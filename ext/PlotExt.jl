module PlotExt

using RomeoDFT
using RomeoDFT: BandsPlotter
using RomeoDFT: ismagnetic, IntersectionMixed, Euclidean, sssp_distance
using DFWannier
using LaTeXStrings
using Plots
using Overseer
using ThreadPools: @tspawnat


gr()
Plots.theme(:wong2)

const E_conv_fac = 13.6056980659


function RomeoDFT.plot_states(es::Vector, nat::Int, gs;
                     include_hub_energy = true,
                     rel_energy = true,
                     fit = false,
                     y_property = "energies",
                     x_property = "magmoms",
                     z_property = "band_distances",
                     base_marker = :circle,
                     base_markersize= 4,
                     base_marker_strokewidth= 2,
                     p = plot(),
                     kwargs...)
                     
    main_es = filter(x->FlatBands in x , es) 
    E_conv_fac = 13.6056980659
    labels = Dict("band_distances" => "\n"*L"\eta\:\textrm{(}n, n_{GS}\textrm{)}",
                  "occupation_distances" => L"d_o\textrm{(}n, n_{GS}\textrm{)}",
                  "energies" => L"\frac{E - E_{GS}}{N_{a}}\, \textrm{(eV)}",
                  "abs_magmoms" => nat > 2 ? L"\sum_I |m^I| \textrm{(}\mu_B\textrm{)}" :
                                   L"|m| \textrm{(}\mu_B\textrm{)}",
                  "magmoms" => nat > 2 ? L"\sum_I m^I \textrm{(}\mu_B\textrm{)}" :
                               L"m \textrm{(}\mu_B\textrm{)}")
    if include_hub_energy
        energies = map(x -> x.total_energy * E_conv_fac / nat, main_es)
    else
        energies = map(x -> dft_energy(x) * E_conv_fac / nat, main_es)
    end
    e_min = include_hub_energy ? gs[Results].total_energy * E_conv_fac / nat : dft_energy(gs) * E_conv_fac / nat 
    properties = Dict()
    properties["energies"]    = rel_energy ? energies .- e_min : energies
    properties["abs_magmoms"] = map(x -> sum(y -> abs(y), x.state.magmoms), main_es)
    properties["magmoms"]     = map(x -> sum(x.state.magmoms), main_es)

    occs = gs[Results].state.occupations
    properties["occupation_distances"] = map(main_es) do e
        x = e.state
        tot = 0.0
        for (o1, o2) in zip(x.occupations, occs)
            d1 = Euclidean()(o1, o2)
            d2 = Euclidean()(o1, DFWannier.ColinMatrix(o2[Down()], o2[Up()]))
            tot += min(d1, d2)
        end
        return tot
    end
    
    minbands = gs[FlatBands].bands
    fermi = gs.fermi
    properties["band_distances"] = map(x -> sssp_distance(x[FlatBands].bands, minbands, fermi), main_es)

    if z_property == "band_distances"
        scatter!(p, properties[x_property], properties[y_property];
                    zcolor = properties[z_property],
                    xguide = labels[x_property],
                    yguide = labels[y_property],
                    colorbar_title = labels[z_property], label = "",
                    size = (600, 300), kwargs...)
    elseif z_property == "none"
        scatter!(p,properties[x_property], properties[y_property];
                    xguide = labels[x_property],
                    yguide = labels[y_property],
                    label = "",
                    size = (600, 300), kwargs...)
    else
        scatter!(p, properties[x_property], properties[y_property];
                    zcolor = properties[z_property],
                    xguide = labels[x_property],
                    yguide = labels[y_property],
                    colorbar_title = labels[z_property], label = "",
                    size = (600, 300), kwargs...)
    end
    if BaseCase in es[end]
        base_e = es[end]
        if Results in base_e && FlatBands in base_e
            res = base_e[Results]
            push!(energies, include_hub_energy ? res.total_energy * E_conv_fac / nat : dft_energy(res) * E_conv_fac / nat)
            push!(properties["energies"], rel_energy ? energies[end] - e_min : energies[end])
            push!(properties["abs_magmoms"], sum(y -> abs(y), res.state.magmoms))
            push!(properties["magmoms"], sum(res.state.magmoms))
                    
            scatter!(p, [properties[x_property][end]], [properties[y_property][end]];  marker=:circle, zcolor = properties[z_property][end],label="", kwargs...)
            scatter!(p, [properties[x_property][end]], [properties[y_property][end]]; color = :red, marker=base_marker, markersize=base_markersize, markerstrokewidth=base_marker_strokewidth, label="vanilla QE")
        end
    end
    return p
end


function RomeoDFT.plot_states(tl::AbstractLedger; unique = false, relaxed = false, unique_thr = 0.1, kwargs...)

    str = tl[Template][1].structure
    nat = length(filter(ismagnetic, str.atoms))
    
    if unique
        es = collect(@entities_in(tl, Unique && Results && FlatBands && !Simulation))
    else
        es = filter(x -> x.converged && (x ∉ tl[Trial] || tl[Trial][x].origin != IntersectionMixed), collect(@entities_in(tl, Results && FlatBands && !Simulation)))
    end
    if relaxed
        es = filter(x -> x ∈ tl[Parent], es)
    else
        es = filter(x -> x ∉ tl[Parent], es)
    end
    if !isempty(tl[BaseCase])
        base_e = tl[entity(tl[BaseCase], length(tl[BaseCase]))]
        if base_e in tl[Results]
            es = [es; base_e]
        end
    end
    gs = tl[ground_state(es)]
    plot_states(es, nat, gs; kwargs...)
end

function plot_animated(tl, dt=0.1; kwargs...)
    str = tl[Template][1].structure
    gs = ground_state(tl)
    nat = length(str.atoms)
    all_es = filter(x->x.converged, collect(@entities_in(tl, Unique && Generation && Results && FlatBands && !Simulation)))
    maxgen = maximum(x->x.generation, tl[Generation])

    mingen = minimum(x->x.generation, tl[Generation])
    p = plot_states(filter(x->x.generation == mingen, all_es), nat, gs; kwargs...)
    display(p)
    for i ∈ mingen+1:maxgen
        es = filter(x->x.generation == i, all_es)
        if isempty(es)
            continue
        end
        plot_states(es, nat, gs; p = p, kwargs...)
        display(p)
        
    end
end

# function plot_states(tl;
#                      include_hub_energy = true,
#                      rel_energy = true,
#                      fit = false,
#                      y_property = "energies",
#                      x_property = "magmoms",
#                      z_property = "band_distances",
#                      p = plot(),
#                      kwargs...)
                     
#     nat = isempty(tl[Template]) ? length(tl[Simulation][1].template_structure.atoms) : length(tl[Template][1].structure.atoms)
#     es = filter(x -> x.converged, collect(@entities_in(tl[Results] && tl[FlatBands] && !tl[Simulation] && !tl[BaseCase])))
#     energies = include_hub_energy ? map(x -> x.total_energy * E_conv_fac / nat, es) :
#                map(x -> dft_energy(x) * E_conv_fac / nat, es)
#     properties = Dict()

#     properties["energies"] = rel_energy ? energies .- minimum(energies) : energies
#     properties["abs_magmoms"] = map(x -> sum(y -> abs(y), x.state.magmoms), es)
#     properties["magmoms"] = map(x -> sum(x.state.magmoms), es)

#     m, mid = findmin(energies)
#     occs = map(x -> x.state, es)
#     properties["occupation_distances"] = map(occs) do x
#         tot = 0.0
#         for (o1, o2) in zip(x.occupations, occs[mid].occupations)
#             d1 = Euclidean()(o1, o2)
#             d2 = Euclidean()(o1, DFWannier.ColinMatrix(o2[Down()], o2[Up()]))
#             tot += min(d1, d2)
#         end
#         return tot
#     end
#     minbands = tl[FlatBands][es[mid]].bands
#     fermi = es[mid].fermi
#     properties["band_distances"] = map(x -> sssp_distance(x.bands, minbands, fermi), es)

#     labels = Dict("band_distances" => "\n"*"eta(n, n_GS)",
#                   "occupation_distances" => "d_o(n, n_GS)",
#                   "energies" => "(E - E_GS)/N (eV)",
#                   "abs_magmoms" => nat > 2 ? "sum(|m_I|) (mu_b)" :
#                                    "|m| (mu_b)",
#                   "magmoms" => nat > 2 ? "sum(m_I) (mu_b)" :
#                                "m (mu_b)")
#     base_id = findfirst(x->x in tl[BaseCase], es)
#     if z_property == "band_distances"
#         scatter!(p, properties[x_property], properties[y_property];
#                     zcolor = properties[z_property],
#                     xguide = labels[x_property],
#                     yguide = labels[y_property],
#                     colorbar_title = labels[z_property], label = "",
#                     size = (600, 300), kwargs...)
#     elseif z_property == "none"
#         scatter!(p,properties[x_property], properties[y_property];
#                     xguide = labels[x_property],
#                     yguide = labels[y_property],
#                     label = "",
#                     size = (600, 300), kwargs...)
#     else
#         scatter!(p, properties[x_property], properties[y_property];
#                     zcolor = properties[z_property],
#                     xguide = labels[x_property],
#                     yguide = labels[y_property],
#                     colorbar_title = labels[z_property], label = "",
#                     size = (600, 300), kwargs...)
#     end
#     if base_id !== nothing
#         scatter!(p, [properties[x_property][base_id]], [properties[y_property][base_id]], color = :red, label="vanilla QE")
#     end
#     return p
# end

function Overseer.update(::BandsPlotter, m::AbstractLedger)
    processed = 0
    simn = simname(m)
    @sync for e in @safe_entities_in(m, SimJob && BandsSettings && TimingInfo && !BandsResults && !Error)
        if processed == 10
            break
        end
        curt = now()
        bp = joinpath(e.local_dir, "bands.out")
        if ispath(bp) && filesize(bp) != 0
            o = Dict()
            @tspawnat 1 try
                plot_path = joinpath(e.local_dir, "bands.png")
                p = plot()
                if e in m[ProjwfcSettings]
                    if ispath(joinpath(e.local_dir, "projwfc.out"))
                        psetting = m[ProjwfcSettings][e]
                        suppress() do
                            return p = plot(local_load(Job(e.local_dir)), e.ymin, e.ymax,
                                            psetting.dos_ratio)
                        end
                    else
                        return
                    end
                else
                    suppress() do
                        return p = plot(local_load(Job(e.local_dir)), bsetting.ymin,
                                        bsetting.ymax; outdat = o)
                    end
                end
                suppress() do
                    return savefig(p, plot_path)
                end
                processed += 1
                m[e] = BandsResults(plot_path)
            catch err
                m[e] = Error(e, err, stacktrace(catch_backtrace()))
            end
        end
        e.postprocessing += Dates.datetime2unix(now()) -
                                           Dates.datetime2unix(curt)
    end
end



export plot_states, plot_animated

end
