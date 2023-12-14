using Flux
using Flux: DataLoader
using Zygote

const HUBTYPE = Vector{Vector{NamedTuple{(:id, :trace, :eigvals, :eigvecs, :occupations, :magmom), Tuple{Int64, NamedTuple{(:up, :down, :total), Tuple{Float64, Float64, Float64}}, NamedTuple{(:up, :down), Tuple{Vector{Float64}, Vector{Float64}}}, NamedTuple{(:up, :down), Tuple{Matrix{Float64}, Matrix{Float64}}}, NamedTuple{(:up, :down), Tuple{Matrix{Float64}, Matrix{Float64}}}, Float64}}}}

function mlp_single_atom(n_features, n_output)
    # model = Chain(Dense(n_features, n_features, x -> leakyrelu(x, 0.2)),
    #               Dense(n_features, n_features, x -> leakyrelu(x, 0.2)),
    #               Dense(n_features, n_features, sigmoid))
    # second = div(n_features, 2)
    # third = div(n_features, 4)
    # model = Chain(Dense(n_features, second, x -> 2 * (sigmoid(x) - 0.5)),
    #               Dense(second, third, x -> 2 * (sigmoid(x) - 0.5)),
    #               Dense(third, second, x -> 2 * (sigmoid(x) - 0.5)),
    #               Dense(second, n_features, x -> 2 * (sigmoid(x) - 0.5)),
    #               )
    # model = Chain(Dense(rand(n_features, n_features).-0.5, true, x -> 2 * (sigmoid(x) - 0.5)),
    #               Dense(rand(n_features, n_features).-0.5, true, x -> 2 * (sigmoid(x) - 0.5)),
    #               # Dense(n_features, n_features, x -> 2 * (sigmoid(x) - 0.5)),
    #               Dense(rand(n_features, n_features).-0.5, true, x -> 2 * (sigmoid(x) - 0.5)),
    #               )
    model = Chain(Dense(n_features, n_features, x -> leakyrelu(x, 0.2)),
                  Dropout(0.2),
                  Dense(n_features, n_features, x -> leakyrelu(x, 0.2)),
                  Dropout(0.2),
                  Dense(n_features, n_output))
    return model
end

function mlp_converge(n_features)
    Chain(Dense(n_features, div(n_features, 2), x -> leakyrelu(x, 0.2f0)),
          Dense(div(n_features, 2), div(n_features, 4), sigmoid),
          Dense(div(n_features, 4), 1, sigmoid))
end

function LinearAlgebra.tr(vm::AbstractVector{RomeoDFT.ColinMatrixType})
    nels = map(vm) do v
        [tr(view(v, Up())), tr(view(v, Down()))]
    end
    reduce(vcat, nels)
end

function mat2features(m::RomeoDFT.ColinMatrixType)
    n = size(m, 1)
    n_half = sum(1:n)
    n_features =n_half *2

    out = zeros(Float32, n_features)
    c = 1
    for i = 1:n
        for j = i:n
            out[c] = m.data[i, j] 
            out[c+n_half] = m.data[i, j+n] 
            c += 1
        end
    end
    out
end

function mat2features(vm::AbstractVector{RomeoDFT.ColinMatrixType})
    n::Int = size(vm[1],1)
    nat::Int = size(vm, 1)
    features = Vector{Float32}(undef, 2nat + nat * (n*(n+1)))
    elements = map(vm) do v
        mat2features(v)
    end
    features[1:2nat] = tr(vm)
    features[2nat+1:end] = reduce(vcat, elements)
    return features
end


function features2mat(v::AbstractVector, nat::Int64)
    n = div(size(v, 1) - 2nat, nat)
    n = Int(div(sqrt(4n+1) - 1, 2))
    m = div(n*(n+1), 2)

    elements = view(v, 2nat+1:length(v))
    
    map(0:nat-1) do iat
        data = zeros(Float32, n, 2n)
        c = 1 + iat*2m
        for i = 1:n
            for j = i:n
                data[i, j] = data[j,i] = elements[c]
                data[i, j+n] = data[j, i+n] = elements[c+m]
                c+=1
            end
        end
        return RomeoDFT.ColinMatrixType(data)
    end
end

function State(model, s::State)
    nat = length(s.occupations)
    State(features2mat(model(mat2features(s.occupations)), nat))
end

@component struct ModelData
    x::Matrix{Float32}
    y::Matrix{Float32}
end

struct ModelDataExtractor <: System end

Overseer.requested_components(::ModelDataExtractor) = (ModelData, Results)

function Overseer.update(::ModelDataExtractor, l::AbstractLedger)
    update_modeldata!(l)
end

function reverse(occs::AbstractVector{RomeoDFT.ColinMatrixType})
    map(occs) do o 
        tmp = similar(o)
        # TODO hardcoding
        tmp.data[1:5, 1:5] = o[Down()]
        tmp.data[1:5, 6:10] = o[Up()]
        return tmp
    end
end

function update_modeldata!(l::AbstractLedger)
    @error_capturing_threaded for e in @entities_in(l, Results && !ModelData)
        path = joinpath(l, e, "scf.out")
        if !ispath(path)
            continue
        end
        
        out = DFC.FileIO.qe_parse_pw_output(path)
        if !haskey(out, :Hubbard)
            continue
        end
        hubbard::HUBTYPE = out[:Hubbard]
        
        r = get(out, :Hubbard_iterations, 1):length(hubbard)-1
        last_state = State(hubbard[end])
        
        states = Vector{State}(undef, length(r))
        Threads.@threads for i in r
            states[i] = State(hubbard[i])
        end

        good_r = findall(x->minimum(x.eigvals[1]) > -0.01 && maximum(x.eigvals[1]) < 1.01, states)
        ngr = length(good_r)
        
        # augment data by flipping the Up and Down channel
        xs = Vector{Vector{Float32}}(undef, 2ngr)
        Threads.@threads for i in 1:ngr
            xs[i] = mat2features(states[good_r[i]].occupations)
            xs[i+ngr] = mat2features(reverse(states[good_r[i]].occupations))
        end

        # filtering only the first half
        isunique = trues(2ngr)
        @inbounds for i in 1:ngr
            if isunique[i]
                xsi = xs[i]
                for j = i+1:ngr
                    if isunique[j]
                        xsj = xs[j]
                        isunique[j] = sum(k -> abs(xsi[k] - xsj[k]), 1:length(xsi)) > 0.1
                    end
                end
            end
        end

        isunique[ngr+1:end] = isunique[1:ngr]
        x = xs[isunique]
        nx = length(x)
        ys = vcat(fill(mat2features(last_state.occupations), div(nx, 2)),
            fill(mat2features(reverse(last_state.occupations)), div(nx, 2)))
        
        # if e.converged
        #     ys = fill(mat2features(last_state.occupations), length(x))
        # else
        #     # not the most performant
        #     ys = fill(0.0f0, length(x))
        # end
        #TODO can we just 1 ys ?
        l[ModelData][e] = ModelData(reduce(hcat, x), reduce(hcat, ys))
    end
end

function prepare_data(l::Searcher)
    es = collect(@entities_in(l, ModelData))
    return prepare_data(es)
end

function prepare_data(es::AbstractVector)
    xs = Matrix{Float32}[]
    ys = Matrix{Float32}[]
    for e in es
        push!(xs, e.x)
        push!(ys, e.y)
    end
    hcat(xs...), hcat(ys...)
end

@pooled_component Base.@kwdef mutable struct TrainerSettings
    n_iterations_per_training::Int
    # This determines how much additional data w.r.t. previous there needs to be, like 1.2= 20% more data
    new_train_data_ratio::Float64
end

@pooled_component struct Model
    n_points::Int
    model_state
end

function model(m::AbstractLedger, i::Int)
    isempty(m[Model]) && return nothing
    nat = length(m[Results][1].state.occupations)
    d = m[ModelData][1]
    n_features = size(d.x, 1)
    n_output = size(d.y, 1)
    tmp_model = mlp_single_atom(n_features, n_output)
    e = entity(m[Model], i)
    Flux.loadmodel!(tmp_model, m[e][Model].model_state)

    X, y = prepare_data(m)
    loss = model_loss(tmp_model(X), y, nat)
    @info loss
    return tmp_model
end

function model_summary(m::AbstractLedger)
    isempty(m[Model]) && return nothing

    nat = length(m[Results][1].state.occupations)
    d = m[ModelData][1]
    n_features = size(d.x, 1)
    n_output = size(d.y, 1)
    tmp_model = mlp_single_atom(n_features, n_output)

    X, y = prepare_data(m)
    data = []
    i = 0
    for (model, es) in pools(m[Model])
        i += 1
        row = Vector{String}(undef, 3)
        row[1] = string(i)
        row[2] = string(model.n_points)
        Flux.loadmodel!(tmp_model, model.model_state)
        loss = model_loss_report(tmp_model(X), y, nat)
        row[3] = string(loss)
        push!(data, row)
    end
    data = permutedims(hcat(data...))

    to_text = sprint() do io
        pretty_table(io, data, header = ["Version", "Training Size", "Loss"])
    end
    return Text(to_text)
end

struct ModelTrainer <: System end
function Overseer.requested_components(::ModelTrainer)
    return (TrainerSettings, Intersection, Model)
end

"""
MSE loss on the full feature set, used for training.
"""
@views function model_loss(y, yhat, nat)
    l1 = Flux.Losses.mse(y[1:2nat, :], yhat[1:2nat, :])
    l2 = Flux.Losses.mse(tanh.(y[2nat+1:end, :]), tanh.(yhat[2nat+1:end, :]))
    return l1 + l2
end

"""
MSE loss only on the occupation matrix, used for evaluation not training.
"""
@views function model_loss_report(y, yhat, nat)
    l2 = Flux.Losses.mse(tanh.(y[2nat+1:end, :]), tanh.(yhat[2nat+1:end, :]))
    return l2
end

function train_model(l::AbstractLedger, n_points)
    trainer_settings = l[TrainerSettings][1]
    X, y = prepare_data(l)
    test, train = Flux.splitobs(size(X,2), at=0.1)
    ndat = size(X[:, train], 2)
    n_features = size(X, 1)
    nat = length(l[Results][1].state.occupations)
    batchsize = min(3000, ndat)

    loader = DataLoader((X[:, train], y[:, train]), batchsize=batchsize, shuffle=true)
    
    model = mlp_single_atom(n_features, n_features)
    opt_state = Flux.setup(Adam(), model)
    
    @info "total training data size $ndat"
    @info "training on batch size $batchsize"

    train_set = [(X[:, train], y[:, train])]
    train_loss = []
    test_loss = []
    min_loss = Inf
    best_state = Flux.state(deepcopy(model))
    for i in 1:trainer_settings.n_iterations_per_training
        for (x, _) in loader
            RomeoDFT.suppress() do
                Flux.train!(model, train_set, opt_state) do m, x, y
                    model_loss(m(x), y, nat)
                end
            end
        end

        push!(train_loss, Flux.Losses.mse(model(X[:, train]), y[:, train]))
        push!(test_loss, Flux.Losses.mse(model(X[:, test]), y[:, test]))
        if i % 100 == 0
            @show i, train_loss[end], test_loss[end]
        end
        if i > 100 && test_loss[end] < min_loss
            best_state = Flux.state(deepcopy(model))
            min_loss = test_loss[end]
        end
    end
    @info min_loss
    
    # p = plot(train_loss, label="train")
    # p = plot!(test_loss, label="test")
    # display(p)
    return Model(n_points, best_state)
end

function Overseer.update(::ModelTrainer, m::AbstractLedger)
    trainer_settings = m[TrainerSettings][1]
    prev_model = isempty(m[Model]) ? nothing : m[Model][end]
    # number of total data points, including duplications
    n_points = sum(x -> x.converged ? x.niterations - x.constraining_steps : 0, m[Results], init=0)
    prev_points = prev_model === nothing ? 0 : prev_model.n_points
    if n_points > prev_points * trainer_settings.new_train_data_ratio
        model = train_model(m, n_points)
        Entity(m, m[Template][1], model, Generation(length(m[Model].c.data)+1))
    end
end

@component Base.@kwdef mutable struct MLTrialSettings
    n_tries::Int = 10000
    minimum_distance::Float64 = 1.0
    use_ml::Bool = true
    prev_model_npoints::Int = 0
end

struct MLTrialGenerator <: System end
function Overseer.requested_components(::MLTrialGenerator)
    return (MLTrialSettings, Model, SearcherInfo)
end

function model(m::AbstractLedger, i=length(m[Model]))
    # if no model yet, skip
    # TODO: Model is a pooled components, and all trials by it would point to the model
    # needs to use pools to deduplicate
    model = isempty(m[Model]) ? nothing : m[Model][i]
    model === nothing && return
    d = m[ModelData][1]
    n_features = size(d.x, 1)
    n_output = size(d.y, 1)
    flux_model = mlp_single_atom(n_features, n_output)
    @debug "loading model with train datasize: $(model.n_points)"
    Flux.loadmodel!(flux_model, model.model_state)
    return flux_model
end

function Overseer.update(::MLTrialGenerator, m::AbstractLedger)
    # TODO: start condition should be after heuristic phase
    # this is something that could be addressed by a higher level system
    if length(m[Results]) < 10
        return
    end

    # if no model yet, skip
    isempty(m[Model]) && return
    model_e = last_entity(m[Model])
    flux_model = model(m)
    romeo_model = m[Model][model_e]
    
    # if model trial has been stopped and no new model, skip
    trial_settings = m[MLTrialSettings][1]
    if !trial_settings.use_ml && trial_settings.prev_model_npoints >= romeo_model.n_points
        return
    end
    trial_settings.prev_model_npoints = romeo_model.n_points
    trial_settings.use_ml = true


    # unique_states = @entities_in(m, Unique && Results)
    # pending_states = @entities_in(m, Trial && !Results)

    n_new = 0
    max_tries = trial_settings.n_tries
    dist_thr  = trial_settings.minimum_distance
    last_max_dist = 0 
    while max_new(m) > 0
        max_dist = 0
        new_s = nothing
        for _ = 1:max_tries
            
            s = State(flux_model, rand_trial(m)[1].state)

            if any(x->any(xi->xi<0, x), diag.(s.occupations)) || any(x -> any(xi->xi < 0, x), s.eigvals)
                continue
            end
        
            min_dist = Inf
            # check duplication with unique results
            for e in @entities_in(m, Unique && Results)
                dist = Euclidean()(e.state, s)
                if dist < min_dist
                    min_dist = dist
                    # min_e = Entity(e)
                end
            end

            # check duplication with previous trials
            for e in @entities_in(m, Trial)
                dist = Euclidean()(e.state, s)
                if dist < min_dist
                    min_dist = dist
                    # min_e = Entity(e)
                end
            end
            
            if min_dist > dist_thr
                max_dist = min_dist
                new_s = s
                break
            else
                # if after max_tries, no trial pass dist_thr, we use trials with max min_dist
                if min_dist > max_dist
                    max_dist = min_dist
                    new_s = s
                end
            end
        end
        @debug "New ml trial with min distance: $max_dist"
        trial = Trial(new_s, RomeoDFT.ModelOptimized) # TODO other tag?
        # add new entity with optimized occ
        new_e = add_search_entity!(m, model_e, trial, m[Generation][model_e])
        m[Model][new_e] = model_e
        n_new += 1
        last_max_dist = max_dist
    end
    if n_new != 0
        @debug "$n_new new ML trials at Generation($(m[Generation][model_e].generation))" 
    elseif max_new(m) > 0 
        @debug "Max reached, max dist = $last_max_dist"
    end
end

## This is super bad
struct MLIntersector <: System end
function Overseer.requested_components(::MLIntersector)
    return (TrainerSettings, Intersection, Model)
end
    
function Overseer.update(::MLIntersector, m::AbstractLedger)
    # if no model yet, skip
    flux_model = model(m)
    flux_model === nothing && return
    model_e = last_entity(m[Model])
    
    unique_states = collect(@entities_in(m, Unique && Results))
    pending_states = collect(@entities_in(m, Trial && !Results))
    nat = length(unique_states[1].state.occupations)

    max_dist = 0
    n_new = max_new(m)
    n_new <= 0 && return
    n_tries = 0
    max_tries = m[MLTrialSettings][1].n_tries
    dist_thr  = m[MLTrialSettings][1].minimum_distance

    lck = ReentrantLock()
    # Trial get the ones that have dist lower than dist_thr
    trial_intersections = Tuple{Float64, Trial, Intersection}[]
    # dist > dist_thr
    good_intersections  = Tuple{Float64, Trial, Intersection}[]
    
    # if length(good) > max_new -> stop and ship it
    Threads.@threads for e1 in unique_states
        if n_new <= 0
            break
        end
        for e2 in unique_states
            if n_new <= 0
                break
            end
            if e1 == e2 
                continue
            end
            for α in (0.25, 0.5, 0.75)
                if n_new <= 0
                    break
                end
                tstate = α * e1.state + (1-α) * e2.state
                tstate = State([features2mat(flux_model(mat2features(tstate.occupations[1])), nat)])
                
                dist, minid   = isempty(unique_states) ? (Inf, 0) : findmin(x -> Euclidean()(x.state, tstate), unique_states)
                dist2, minid2 = isempty(pending_states) ? (Inf, 0) : findmin(x -> Euclidean()(x.state, tstate), pending_states)
                lock(lck) do
                    dist3, mid3   = isempty(good_intersections) ? (Inf, 0) : findmin(x -> Euclidean()(x[2].state, tstate), good_intersections)
                    dist = min(dist, dist2, dist3)
                end


                if dist > dist_thr
                    lock(lck) do
                        push!(good_intersections, (dist, Trial(tstate, ModelOptimized), Intersection(Entity(e1), Entity(e2))))
                        n_new -= 1
                    end
                else
                    lock(lck) do
                        push!(trial_intersections, (dist, Trial(tstate, ModelOptimized), Intersection(Entity(e1), Entity(e2))))
                    end
                end
            end
        end
    end
    while n_new > 0 && !isempty(trial_intersections)
        sort!(trial_intersections, by = x -> min(x[1], minimum(y->Euclidean()(x[2].state, y[2].state), good_intersections, init=Inf)))
        push!(good_intersections, pop!(trial_intersections))
        n_new-=1
    end
    for (_, trial, intersection) in good_intersections 
        new_e = add_search_entity!(m, model_e, trial, m[Generation][model_e], intersection)
        m[Model][new_e] = model_e
    end
    @debug "$(length(good_intersections)) new ML trials at Generation($(m[Generation][model_e].generation))" 
end

## Save for later
# abstract type AbstractSettings end
# @component struct HighLevelSettings <: AbstractSettings
#     #
#     #
#     # 
# end

# @component struct HighLevelResults
# end

# result_comp(::Type{HighLevelSettings}) = HighLevelResults

# struct HighLevel <: System end

# function Overseer.update(::HighLevel, l::Searcher)
    
#     for e in @entities_in(l, HighLevelSettings)
        
#         if !(e in l[SCFSettings])
#             l[e] = SCFSettings(e.create_scf_settings)
#         elseif e in l[SCFSettings] && e in l[Results] && !(e in l[NSCFSettings])
#             # do something with e
#             l[e] = NSCFSettings(e.create_nscf_settings)
#         elseif e in l[NSCFSettings] && e in l[NSCFResults] && !(e in l[BandsSettings])
#             # do something with nscf output
#             l[e] = BandsSettings(e.create_bands_settings)
#         elseif e in l[BandsResults]
#             l[e] = HighLevelResults(x, y, z)
#         end
#     end
# end

# struct DoneChecker <: System end

# function Overseer.update(::DoneChecker, l::Searcher)
#     dones = Dict()
#     done_c = l[Done]
#     for c in components(l, AbstractSettings)
#         done_comp = l[result_comp(eltype(c))]

#         for e in @entities_in(c && !done_c)
#             dones[Entity(e)] = get!(dones, Entity(e), true) && e in done_comp
#         end
#     end

#     for (e, d) in dones
#         if d
#             done_c[e] = Done(false)
#         end
#     end
# end
