struct RandomTrialGenerator <: System end
function Overseer.requested_components(::RandomTrialGenerator)
    return (RandomSearcher, Intersection, BaseCase)
end

function rand_trial(l::Searcher, n=1)
    base_e     = entity(l[BaseCase], 1)
    base_state = l[Results][base_e].state
    nelec      = round.(Int, base_state.totoccs)
    norb       = size.(base_state.occupations, 1)

    # Here we check whether different oxidation states have been tried
    # if yes we start randomly taking the total occupations for eigenvalues
    ox = isempty(l[Trial]) ? [0,0] : extrema(x->sum(x.state.totoccs), l[Trial])
    diff_ox_tried = (ox[1] != ox[2])

    out = Trial[rand_trial(norb, nelec) for _ in 1:n]
    
    # for i = 1:n
    #     if !diff_ox_tried
    #         push!(out, rand_trial(norb, nelec))
    #     else
    #         # take eigenvalues from different entities
    #         eigvals = map(1:length(nelec)) do i
    #             # e = Entity(l[Unique], rand(1:length(l[Unique])))
    #             # while !(e in l[Results])
    #             #     e = Entity(l[Unique], rand(1:length(l[Unique])))
    #             # end
    #
    #             e = rand(collect(@entities_in(l, Unique && Results)))
    #             l[Results][e].state.eigvals[i]
    #         end
    #         
    #         push!(out, rand_trial(eigvals))
    #     end
    # end
    return out
end

function rand_trial(eigvals::Vector)
    n_orb = div(length(eigvals[1]), 2)
    nangles = div(n_orb * (n_orb - 1), 2)
    
    rand_angles() = Angles([π * (rand() - 0.5) for i in 1:nangles])
    
    occs = map(eigvals) do eig
        D = DFWannier.MagneticVector(eig)
        V = DFWannier.ColinMatrix(Matrix(rand_angles()), Matrix(rand_angles()))
        return Matrix(Eigen(D, V))
    end
    return Trial(State(occs), RandomMixed)
end
    
function rand_trial(n_orb_per_at::Vector, n_elec_per_at::Vector)
    
    occs = map(zip(n_orb_per_at, n_elec_per_at)) do (norb, nelec)
        nangles = div(norb * (norb - 1), 2)
        
        rand_angles() = Angles([π * (rand() - 0.5) for i in 1:nangles])
        
        ox_state_offset = rand([-1, 0, 1])
        diagvec = zeros(2norb)
        
        while sum(diagvec) < min(nelec + ox_state_offset, 2norb)
            diagvec[rand(1:2norb)] = 1.0
        end
        
        if norb == 1
            return ColinMatrixType(diagm(0 => diagvec[1:1]), diagm(0 => diagvec[2:2]))
        else
            D = DFWannier.MagneticVector(diagvec)
            V = DFWannier.ColinMatrix(Matrix(rand_angles()), Matrix(rand_angles()))
            return Matrix(Eigen(D, V))
        end
    end

    return Trial(State(occs), RandomMixed)
end

function Overseer.update(::RandomTrialGenerator, m::AbstractLedger)
    # should we throw error if either is empty?
    if isempty(m[RandomSearcher]) || isempty(m[BaseCase])
        @error "Error in searcher initialization, exit RadomTrialGenerator without creating trials"
        return
    end

    # check if there is still random search budget
    # should be redundant, should be done in core loop
    random_search = singleton(m, RandomSearcher)
    n_random = length(filter(e->e.origin==RandomMixed, m[Trial]))
    if n_random >= random_search.nsearchers
        return
    end

    # at least one base case calculation is finished
    base_e = filter(@entities_in(m, BaseCase)) do e
        all_children_done(m, e) && e ∈ m[Results] && !isempty(m[Results][e].state.occupations)
    end
    if isempty(base_e)
        @debug "Base cases not finished, exit RadomTrialGenerator without creating trials"
        return
    else
        base_e = base_e[1]
    end
    
    rand_search_e = entity(m[RandomSearcher], 1)

    maxgen = maximum_generation(m)
    n_new = max_new(m)
    for trial in rand_trial(m, n_new)
        e = add_search_entity!(m, rand_search_e,
                               trial,
                               Generation(maxgen))
                               
        if Hybrid in m && length(m[Hybrid]) != 0
            m[e] = Hybrid()
        end
        n_new += 1
        
    end
    if n_new != 0
        @debug "$n_new random trials at Generation($(maxgen))."
    end
end
