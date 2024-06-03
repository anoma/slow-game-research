# Copyright 2024 Heliax AG
# SPDX-License-Identifier: GPL-3.0-only

module SlowGameSim

using Distances
using Distributions
using Random
using StatsBase
using LinearAlgebra
using TensorCore
using AxisArrays

using OptimalTransport
using Tulip

using GLMakie
GLMakie.activate!()

# Needs to happen last, to override Axis
import AxisArrays: Axis as Axis


# Example:
# Two player thermostat
# Slow regulator decides on a policy, which gives a permissible range of temperatures.
# Fast operator receives a budget to heat/cool the office to temperaturs within that range.
# Outside temperature fluctuates, necessitating heating and cooling, which both incur a cost.
# The operator can keep any part of the budget that was not spent at the end of the period.
####


####
# Fast player/ operator can set the Thermostat
# Regulator/Slow player sets a policy of permitted temperatures
# 1) look at the setting
# 2) measure room temp (this is instantaneous for now)
#
# Policy: stay within room temps [x,y]
#
# Upon detection of violation (or above a certain threshold of certainty), slow player can enforce some additional cost.
#
# Both of these can
# a) happen at different relative frequencies
# b) happen during the interval to be measured or post-hoc analyzing recorded data
#
# To compute the estimates, we implement a model inspired by
# Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff https://arxiv.org/abs/1901.07821

# Later:
#   - operators should be incentivized to provide accurate historical data/information about their capabilities to the regulators, s.t. the regulator
#     can produce accurate estimates of detection probability. regulators could also operate parts of the service to gather this data themselves
#

# Model:
# Fast operator
#   - operates the heating
#   - room temperature is a public channel on which messages (temp updates) are posted
#   - the room can be set to integrate temperature changes over the last N steps, to clarify the distinction between actions and observables
#
# Slow regulator
#   - computational and memory limitations in relation the the fast actors lead to dropped messages
#   - above limitations are caused e.g. by internal consensus requiring multiple rounds of messages and a
#     constant factor on resources to agree on the state of the channel/each message on it
#   - not the individual agents constituting the slow player, but the game in aggregate is slow
#

# Noise:
#   - a slow actor is a special case of a low capacity actor

# Dropout:
#   - regulator records everything, but only has memory for N messages
#   - if analysis happens every M messages and M > N, regulator needs to drop messages
#     e.g. equidistant sampling at M = 2N would be to save every second message


# Knowledge Assumptions for Regulators:
# - regulators can simulate in-policy behavior and measure divergence of observations from that
# - regulators know how lossy their measurments are
# - optional: regulators have some knowledge about the specifics of cheating. TODO: simulate different degrees of knowledge, not just full knowledge
#
# Knowledge Assumptions for Operators:
# - operators can have an agent in the slow player to know, or at least approximate the measurement error

# Model components from Lossy Compression Model (https://arxiv.org/abs/1901.07821)
#   - signal
#   - encoder
#   - decoder
#   - (perceptual) divergence
#   - estimated signal
#   - (TODO?: distortion)
#
# Our addition
#   - interpolation during decoding

# Goals for incentive structure
#   - regulators set policy and want operator to adhere to it
#   - operator wants to minimize incurred cost / maximize reward
#   -> defection from policy should be more costly in expectation than adhering to it
#
# Resulting game
#   - regulator(s) and operator(s) modelled by one player each
#   - regulator R measures operator O behavior
#   - since memory and bandwidth of R are limited R, it can not preserve full information
#       - i.e. for measurement rate M of R, action rate A of O, interval I and 2M = A, R can only preserve half the information from I
#   - operator wants to prevent detection of defection to not incur enforcement cost
#
# Parameters influencing divergence of signal and estimate:
#   - Noise + type of interpolation determine divergence iff encoding is 1:1 and decoding is lossy
####


# actions
# One agent is determined to be in charge of the heater for the room.
# The heater has no built in thermostat and the task of the operator is to regulate it in a way a thermostat would.
# (E.g. the agent can measure room temp and decide to cool or heat by a certain amoint)

# randomize outside temperature within some bounds
function outside_temp(l, u, length)
    # TODO: maybe smooth this
    out = rand(DiscreteUniform(l, u), length)

    return out
end

# compute necessary temperature correction to move inside temperature to within policy bounds [p_l, p_u]
# assume operator has a sensor to measure outside temp without lag
# heating or cooling by 1 incurs cost of 1
function temp_correct(p_l, p_u, temps)::Vector{Int64}
    output = zeros(length(temps))

    for (i, temp) in enumerate(temps)
        if temp > p_u
            output[i] = temp - p_u
        elseif p_l < temp
            output[i] = p_l - temp
        end
    end
    return output
end

# cheat with probability p and "amplitude" a
function cheat(p, a, correction)
    noise = rand(Binomial(a, p))
    #noise = round(rand(Binomial(a, p)) * rand(Pareto(10,10)))

    #if p == 0
    #    shape = 1
    #else
    #    shape = 20 / (20 * p)
    #end

    # TODO: Add an option for a heavy tailed dist.
    #       widen temperature range to increase reward potential and make it worth it.

    #noise = round(rand(Pareto(shape, a)))


    if sign(correction) == 1
        if noise >= correction
            return 0
        else
            return correction - noise
        end
    elseif sign(correction) == -1
        if abs(noise) >= abs(correction)
            return 0
        else
            return correction + noise
        end
    else
        return 0
    end
end
# Note: With binomial noise, measurement error and cheating seem to cancel each other out in high error/high cheat regimes, s.t. divergence goes down again.
# This is taken into account later when choosing wether to derive reward structure from data including or exluding priors about interaction of measurement error and cheating noise.

####
# signal
# X
# generate a time series of temperature data points, radmonized within [u,l]
function generate_signal(t_l, t_u, p_l, p_u, length, p, a)
    temps = outside_temp(t_l, t_u, length)
    corrections = temp_correct(p_l, p_u, temps)
    actions = map(x -> cheat(p, a, x), corrections)
    return actions
end


function downsample(integrate_steps, signal)
    @assert(mod(length(signal), integrate_steps) == 0) # signal should be divisible by integrate_steps

    return collect(map(sum, Base.Iterators.partition(signal, integrate_steps)))
end

# Encoder
# encodes from action(s) to observable
# To model the temperature lag of the room, integrate_steps can be used to denote the period of aggregating actions.
# This can be used to emphasize the distinction between actions and observables.
function encode(integrate_steps, signal)
    return downsample(integrate_steps, signal)
    # add no randomness to temp lag of room, s.t. fast actor does not need to measure, only set cooling/heating
    # Alternative: add measurements for the fast actor
end

# This implements an error, modeled by stochastic dropout
function stochastic_dropout(signal, error)
    # error = float in [0, 1] and denotes be chance to miss a value (dropout)
    @assert(0 <= error <= 1)

    output = Array{Int64}(undef, length(signal))

    for i in 1:length(signal)
        noise = rand(Bernoulli(error))

        if noise == 0
            output[i] = typemin(Int64) # failure -> value set to special value that should always be out of range
        else
            output[i] = signal[i]
        end
    end
    output
end

# Dropout depends on speed factor (sf) of the regulator, which results in a difference in capacity.
# At speed ratio 1:1 the sf is 1, at speed ratio 2:1 (i.e. the operator is twice as fast as the regulator), sf is 0.5.
# This models a regulator that is only able to make some measurements, or store less data than is produced by the operator in an interval.
function dropout(signal, speed_factor)
    @assert(0 <= speed_factor <= 1)
    # here we assume the fast and slow agents use the same technique, but at different resolutions
    output = Array{Int64}(undef, length(signal))

    # The following implements a rougly equidistant dropout pattern
    # TODO: Right now the last interval is cut short, take that into account and shift the gap positions
    samples = length(signal) * speed_factor
    gaps = length(signal) - samples
    gap_position = length(signal) / gaps

    for i in 1:length(signal)
        # since we're doing modular arithmetic with floats, we check if i < x < i+1 s.t. mod(x, gap_position) == 0, i integer and x float
        if (mod(i+1, gap_position) > mod(i, gap_position))
            output[i] = signal[i]
        else
            output[i] = typemin(Int64)
            # drop -> value set to special value that should always be out of range
        end
    end
    return output
end

function interpolate(signal) # Note: Different implementations of this function should result in different measurement errors
    output = zeros(length(signal))

    # assuming we get a signal with missing elements, we take the mean of non-NaN values
    mean = round(StatsBase.mean(filter(x -> x!= typemin(Int64), signal)), RoundNearestTiesAway)

    if isnan(mean) # in case there are only dropout values, mean would be NaN so we set it to 0
        mean = 0
    end # Note: regular comparison (==) does not work

    for i in 1:length(signal)
        if signal[i] == typemin(Int64)
            output[i] = mean
        else
            output[i] = signal[i]
        end
    end

    return output
end

# decoder
# decodes from an observation to an estimate of an observable (which could imply an estimated action, but does not have to)
# read setting, read temperature; both with frequency M
function decode(signal, e)
    noisy = dropout(signal, e)
    output_estimate = interpolate(noisy)
    return output_estimate
end

# estimated signal
# X' (\hat{X} in paper)
# estimate = decode(encode(signal), e)

# perceptual divergence
# d(p_X, p_X')
# e.g. Kulback-Leiber (KL), Wasserstein, or use case specific perceptive divergence
# We are using wasserstein distance, since it is a metric, does not give us singularities anywhere and is symmetric
function divergence(signal, estimate)
    d = ot_cost(sqeuclidean, signal, estimate)
    return d
end


# Here we do a simplistic mapping of timeseries to a distribution by aggregating settings by temperature, forgetting time information
# since our policy only defines a permitted range and no time dependence.
# Note: l and u need to be the lower and upper bounds of all data that will be mapped across all experiments in a run
function vecs_to_dist(vectors, l, u, scale)
    acc_size = u+abs(l)+1 # number of buckets
    acc = zeros(Int64, acc_size)

    for vector in vectors
        addcounts!(acc, vector, (l:u))
    end

    p_vec = Vector{Float64}(undef, acc_size)

    for i in (1:acc_size)
        p_vec[i] = acc[i] / scale
    end

    return DiscreteNonParametric((l:u), p_vec)
end

# TODO: refactor
function vec_to_dist(vector, l, u, scale)
    acc_size = u+abs(l)+1 # number of buckets
    acc = zeros(Int64, acc_size)

    addcounts!(acc, vector, (l:u))

    p_vec = Vector{Float64}(undef, acc_size)

    for i in (1:acc_size)
        p_vec[i] = acc[i] / scale
    end

    return DiscreteNonParametric((l:u), p_vec)
end

# Option: we could preserve timing information by bucketing by (time, setting)

function add_noise_to_dist(l, u, ratio, dist) # Dist needs to be DiscreteNonParametric
    noise = rand(DiscreteUniform(l, u), length(probs(dist))) * ratio
    if ratio != 1
        noise = map(x -> round(Int, x), noise)
    end

    noise_dist = vec_to_dist(noise, l, u, length(probs(dist)))
    xs, ps = params(dist)
    DiscreteNonParametric(xs, (ps+probs(noise_dist))/2)
end

# This function computes the expected cost of an action distribution
# Heating or cooling costs are 1:1 correlated to the magnitude of the change
function compute_expected_cost(dist, sample_size)
    # Cost is 1:1 to absolute values
    xs, probs = params(dist)

    cost = dot(map(abs, xs), probs) * sample_size
    cost
end

function compute_bucketing_bounds(sig_acc, est_acc)
    min_s = minimum(map(minimum, sig_acc))
    min_e = minimum(map(minimum, est_acc))
    max_s = maximum(map(maximum, sig_acc))
    max_e = maximum(map(maximum, est_acc))

    if min_s <= min_e
        l = min_s
    else
        l = min_e
    end
    if max_s >= max_e
        u = max_s
    else
        u = max_e
    end

    n_buckets = u+abs(l)+1

    return (l, u, n_buckets)
end


function experiment(error, cheat_p, cheat_a, epochs, sample_size, integrate_steps)
    # Bounds regulator has decided upon as target temperature range for a room.
    policy_lower = 18
    policy_upper = 25

    # the bounds between which outside temperature can fluctuate
    outside_temp_min = 10
    outside_temp_max = 32

    integrated_sample_size::Int64 = sample_size / integrate_steps
    p_scale = epochs * integrated_sample_size

    sig_acc = Array{Vector{Int64}}(undef, epochs)
    est_acc = Array{Vector{Int64}}(undef, epochs)

    # Error = 1 - speed factor
    for i in 1:epochs
        signal = generate_signal(outside_temp_min, outside_temp_max, policy_lower, policy_upper, sample_size, cheat_p, cheat_a)
        sig_acc[i] = encode(integrate_steps, signal)
        est_acc[i] = decode(sig_acc[i], 1 - error)
    end

    l, u, _ = compute_bucketing_bounds(sig_acc, est_acc)
    scaled_l = integrate_steps * l
    scaled_u = integrate_steps * u
    sig_dist = vecs_to_dist(sig_acc, scaled_l, scaled_u, p_scale)
    est_dist = vecs_to_dist(est_acc, scaled_l, scaled_u, p_scale)

    sig_cost = compute_expected_cost(sig_dist, sample_size) # takes sample_size since intervals are the same length

    d = divergence(sig_dist, est_dist)
    return (d, sig_cost, sig_dist, est_dist)
end

# Generate a coordinate grid
function gen_grid(a, b, n)
# TODO: catch all corner cases (division by 0 etc.) for more flexbility
    h = round( (b-a)/(n-1), digits=2)
    collect(a:h:b)
end

# error_batches determines # of batches varied by error
# cheat_batches determines # of batches varied by cheat
# prior_ratio determines amount of knowledge about the cheating / error interaction
function experiments(error_batches, cheat_batches, integrate_steps_max, cheat_a_max, epochs, s_size)
    # TODO?: cheat_a_res

    # Initialize an AxisArray of the correct shape and copy it for conciseness.
    # TODO ARRAY: *_batches does not include 0 batch
    A = AxisArray(zeros(error_batches+1, cheat_batches+1, cheat_a_max); # zeros(cheat_a) zeros(integrate_steps_max)],
        speed_factors = (gen_grid(0, 1, error_batches)),
        cheat_ps = (gen_grid(0, 1, cheat_batches)),
        cheat_as = 1:cheat_a_max,
        )


    # TODO: add dims and loop for: integrate_steps, partial_priors
    sig_est_divs = deepcopy(A) # divergence of signal and estimate
    tru_cheat_divs = deepcopy(A) # divergence between non-cheat estimate and cheat estimate
    sig_costs = deepcopy(A) # cost of each signal

    # TODO: rewrite with iterators or transducers for more modularity
    for (i, sf) in enumerate(A[Axis{:speed_factors}]) # TODO: start at error and cheat 0
        _, _, comp_sig, comp_est = experiment(sf, 0, 0, epochs, s_size, integrate_steps_max) # to store the non-cheating signal for each cheating batch
        comp_div_cutoff = divergence(comp_sig, comp_est)
        for (j, cheat_p) in enumerate(A[Axis{:cheat_ps}])
            for (k, cheat_a) in enumerate(A[Axis{:cheat_as}])
                # TODO: add one more loop
                #for (l, prior_ratio) in enumerate(A[Axis{:cheat_as}])

                    axis = [Axis{:speed_factors}(i), Axis{:cheat_ps}(j), Axis{:cheat_as}(k)] # TODO: can we do this more elegantly?

                    div, cost, _, estimate = experiment(sf, cheat_p, cheat_a, epochs, s_size, integrate_steps_max)

                    # "..." unpacks args from array
                    if j == 1
                      sig_est_divs[axis...] = div
                    else
                      sig_est_divs[axis...] = div #+ rand(Normal(0, 1)) # TODO: Make this a parameter in the experiment call # TODO: vanilla codepath option
                    end

            #partial_prior = add_noise_to_dist(-7, 8, estimate)
            # TODO: to simulate partial priors about the interaction of noise and cheating, we could add (stochastic) dropout + interpolation
            #       in compare and estimate and see the influence it has on the relation between se_div and cheat_div, as well as cheat_div itself
            # TEST: in the limit with increasing droput, se_divs and only_error should converge

                # Measure divergence of estimated signal while cheating from signal while being truthful
                tru_cheat_divs[axis...] = divergence(comp_sig, estimate) - comp_div_cutoff
                sig_costs[axis...] = cost
            end
        end
    end
    return (sig_est_divs, sig_costs, tru_cheat_divs)
end



function surface_plot_util(fig, viz_state, axis_util, data, data_label, data_title)
    (x_label, y_label, x_data, y_data, _) = axis_util

    # TODO: unfold viz_state if code changes s.t. it contains more than a single value

    x_indices = 1:length(x_data)
    y_indices = 1:length(y_data)

    if data isa AxisArray
    else
        data = to_aa(data)
    end

    surface_data = @lift(data[Axis{:speed_factors}(:), Axis{:cheat_ps}(:), Axis{:cheat_as}($viz_state)].data)#, Axis{:integrate_steps}(i)

    xt = (x_indices[begin:2:end], map(string, x_data[begin:2:end]))
    yt = (y_indices[begin:2:end], map(string, y_data[begin:2:end]))

    ax = Axis3(fig, xlabel = x_label, ylabel = y_label, zlabel = data_label,
            xticklabelsize = 12, yticklabelsize = 12,
            zticklabelsize = 10, zlabeloffset = 50,
            xticks = xt, yticks = yt,
            title = data_title
        )

    @lift begin
        empty!(ax)
        Makie.surface!(ax, x_indices, y_indices, $surface_data)
    end
end

function plot_interaction(fig, viz_state, sg)
    # TODO: pass bounds via state var
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press || event.action == Keyboard.repeat
            if event.key == Keyboard.up && to_value(viz_state) < 10
                viz_state[] = viz_state[] + 1
            elseif event.key == Keyboard.down && to_value(viz_state) > 1
                viz_state[] = viz_state[] - 1
            else
            end
        end
        set_close_to!(sg.sliders[1], to_value(viz_state[]))
    end

    on(sg.sliders[1].value) do val
        viz_state[] = val
    end

    return (viz_state, fig)
end

function normalize_reward(reward)
    norm_reward = zeros(size(reward))

    # take mean along each error_regime and subtract that
    for i in 1:length(reward[Axis{:speed_factors}])
        for k in 1:length(reward[Axis{:cheat_as}])
        mean = StatsBase.mean(reward[Axis{:speed_factors}(i), Axis{:cheat_ps}(:), Axis{:cheat_as}(k)])
        if mean < 0 # set mean to 0, since we only want to curtail subsidies
            mean = 0
        end
            for j in 1:length(reward[Axis{:cheat_ps}])
                norm_reward[i, j, k] = reward[i, j, k] - mean
            end
        end
    end
    to_aa(norm_reward)
end

# compute regret along cheat_p axis
function compute_regret(reward)
    regret = zeros(size(reward))
    for i in 1:length(reward[Axis{:speed_factors}])
        for k in 1:length(reward[Axis{:cheat_as}])
        max_reward = maximum(reward[Axis{:speed_factors}(i), Axis{:cheat_ps}(:), Axis{:cheat_as}(k)])
            for j in 1:length(reward[Axis{:cheat_ps}])
                regret[i, j, k] = max_reward - reward[i, j, k]
            end
        end
    end
    regret
end

function to_aa(a)
  AxisArray(a, :speed_factors, :cheat_ps, :cheat_as)
end

# TODO: build a utility that just takes the AxisArray and then generates all the rest

# TODO: plot cheat_a, integrate_steps, partial_priors
function experiment_and_plot(error_batches, cheat_batches, integrate_steps, cheat_a_max, epochs, sample_size, save)
    (se_divs, s_costs, cheat_divs) = experiments(error_batches, cheat_batches, integrate_steps, cheat_a_max, epochs, sample_size)
    x_ax, y_ax, z_ax = axisvalues(se_divs)
    axis_util = ("(1 - speed factor) of regulator", "P(cheat)", x_ax, y_ax, z_ax)

    viz_state_init = cheat_a_max
    viz_state = Observable{Int64}(viz_state_init)
    fig = Figure(size = (1500, 1500))

    if save == 0
        slider = SliderGrid(fig[0, 1], (label = "cheat amplitude", range = 1:1:10, format = "{1:d}", startvalue = viz_state_init))
    else
    end
    # TODO: add one more slider

    # se_divs contains priors about interaction of cheating and measurement error
    # cheat_divs contains priors about connections of non-cheating and cheating estimated signals
        # regulators can generate this data by computing estimates of (a large enough set) of non-cheating signals and computing divergence to whatever they observe
        # this way, we should be able to use cheat_divs as an example without assuming any further priors
    # slices along the error axis should give us punishment curve

    budget = 4 * sample_size # TODO: improve budgeting model
    reward = to_aa(fill(budget, error_batches+1, cheat_batches+1, cheat_a_max) - s_costs) # reward before policy enforcement

    only_error_vec = se_divs[Axis{:speed_factors}(:), Axis{:cheat_ps}(1), Axis{:cheat_as}(1)]
    only_error = repeat(only_error_vec, 1, cheat_batches+1, cheat_a_max) # measurement error only

    # punish_*_priors can be seen as excess divergence not explained by measurement error
    punish_no_priors = only_error - cheat_divs
    punish_with_priors = se_divs - cheat_divs

    reward_no_priors = normalize_reward(to_aa(hadamard(reward, punish_no_priors)))
    #reward_no_priors = to_aa(hadamard(reward, punish_no_priors)) needs more subsidies but leads to same regret structure
    reward_with_priors = normalize_reward(to_aa(hadamard(reward, punish_with_priors)))

    pure_regret = compute_regret(reward)
    regret_no_priors = compute_regret(reward_no_priors)
    regret_with_priors = compute_regret(reward_with_priors)

    # INCENTIVE MODEL
    #
    # We assume if we know the noise of our measurements, and the resulting divergence of estimates to signals, it corresponds
    # to the maximum divergence of other in-policy signals. Thus, any observed estimate that diverges more from our in-policy simulations
    # is deemed out of policy.
    #
    # To model the influence our measurement error has on our estimates, we simulate signals, reconstruct estimates using the lossy measurements we have access to
    # and compute d_e = div(signal, estimate), for error e.
    #
    # Later, if we know the measurement error during an interval, we can compute d_o = div(estimate of observed signal, known good signal).
    # First order incentive structure approximation: To get a reward factor, we can compute (d_e - d_o) to receive a weight for the reward.
    #
    # There are two settings:
    #
    # 1) We have no priors about cheating (use only_error)
    # Divergence up to only_error needs to be permitted for each error regime.
    # Excess divergence used as direct weight
    #
    # 2) We have priors about the interaction of cheating noise and measurement error (use se_divs)
    # Divergence can be more tightly bounded to be up to only se_divs for each error regime

    # IN ALL CASES, MEASUREMENT ERROR MUST BE KNOW TO SET REWARD CURVE

    # Note: A reward structure s.t. no-cheating incurs the smallest total cost in all error regimes would be DSIC
    # TODO: simulate settings with partial priors (right now, only full knowledge)

    if save == 0
        viz_state, fig = plot_interaction(fig, viz_state, slider)
    else
    end

    surface_plot_util(fig[1,2], viz_state, axis_util, only_error, "error induced div w/o priors on cheating", "1.2")
    surface_plot_util(fig[1,3], viz_state, axis_util, se_divs, "div w/ priors on cheating/error interaction", "1.3")

    surface_plot_util(fig[2,1], viz_state, axis_util, cheat_divs, "excess div of estimate from good baseline", "2.1")
    surface_plot_util(fig[2,2], viz_state, axis_util, punish_no_priors, "reward weighting factor w/o priors", "2.2")
    surface_plot_util(fig[2,3], viz_state, axis_util, punish_with_priors, "reward weighting factor w/ priors", "2.3")

    surface_plot_util(fig[3,1], viz_state, axis_util, reward, "pure reward (remaining budget)", "3.1")
    surface_plot_util(fig[3,2], viz_state, axis_util, reward_no_priors, "adjusted reward w/o priors", "3.2")
    surface_plot_util(fig[3,3], viz_state, axis_util, reward_with_priors, "adjusted reward w/ priors", "3.3")

    surface_plot_util(fig[4,1], viz_state, axis_util, pure_regret, "regret w/o enforcement", "4.1")
    surface_plot_util(fig[4,2], viz_state, axis_util, regret_no_priors, "regret w/o priors", "4.2")
    surface_plot_util(fig[4,3], viz_state, axis_util, regret_with_priors, "regret w/ priors", "4.3")

    fig
end

# TODO: simulate randomized measurements by regulator and how they change operator regret 

# TODO: simulate case when we only have partial information about in-policy baseline

# TODO: simulate case where extraction opportunity is distributed in a heavy tailed way, e.g. pareto
#       i.e. its worth it for the attacker to almost never cheat, just in very few cases with high reward
#       Can we find use cases, or a general approach, to make a system self tuning, i.e. aggregate heavy tailed cases into subgaussions?
#       Would good priors help? How accurate would they need to be to be sufficient?

# TODO: refactor s.t. that we can compose noise, prior and action distributions more easily


# EXAMPLE
# 1-10 epochs are informative
# 1, 2, 5, 10 are possible values for integrate_steps
experiment_and_plot(20, 20, 1, 10, 1, 10000, 0)
#experiment_and_plot(20, 20, 1, 10, 10, 1000, 0)

# Uncomment and execute the next two lines for saving figures to png
#plot = experiment_and_plot(20, 20, 1, 3, 1, 10000, 1)
#save("fig.png", plot)

end # module SlowGameSim
