# File that generates DDE dynamics of several systems

using DifferentialEquations
using DelimitedFiles
using PyPlot

# Basic check 1 =========================================
function basic_check_1(du, u, h, p, t)
    tau = p
    hist1 = h(p, t-tau)[1]
    du[1] =  u[1] * (1 - hist1)
end

h(p,t) = 1.2 * ones(1)   
u0 = [1.2]
tau = 1.0
lags = [tau]
p = (tau)
tspan = (0.0, 50.0)


prob = DDEProblem(basic_check_1, u0, h, tspan, p; constant_lags=lags)
alg = MethodOfSteps(Tsit5()) # doesn't work with DP5 DP8 but works with Tsit5 and Bosh3
sol = solve(prob,alg, saveat=0.05)
usol = transpose(hcat(sol.u...))
time = sol.t

plot(sol.t, usol)
plt.xlabel("Time")
plt.show()

writedlm("test_basic_check_1.txt", [sol.t usol])

# Basic check 2 =========================================
function basic_check_2(du, u, h, p, t)
    tau = p
    hist1 = h(p, t-tau)[1]
    du[1] =  u[1] * (1 - hist1)
end

h(p,t) = 1.2 * ones(1)   
u0 = [1.2]
tau = 2.0
lags = [tau]
p = (tau)
tspan = (0.0, 50.0)


prob = DDEProblem(basic_check_2, u0, h, tspan, p; constant_lags=lags)
alg = MethodOfSteps(BS3()) # doesn't work with DP5 DP8 but works with Tsit5 and Bosh3
sol = solve(prob,alg, saveat=0.05)
usol = transpose(hcat(sol.u...))
time = sol.t

plot(sol.t, usol)
plt.xlabel("Time")
plt.show()

writedlm("test_basic_check_2.txt", [sol.t usol])


# Basic check 3 & 4 =========================================
function basic_check_3(du, u, h, p, t)
    tau = p
    hist1 = h(p, t-tau)[1]
    du[1] =  u[1] * (1 - hist1)
end

h(p,t) = 1.2 * ones(1)   
u0 = [1.2]
tau = 3.0
lags = [tau]
p = (tau)
tspan = (0.0, 50.0)


prob = DDEProblem(basic_check_3, u0, h, tspan, p; constant_lags=lags)
alg = MethodOfSteps(BS3()) # doesn't work with DP5 DP8 but works with Tsit5 and Bosh3
sol = solve(prob,alg, saveat=0.05)
usol = transpose(hcat(sol.u...))

alg = MethodOfSteps(Tsit5()) # doesn't work with DP5 DP8 but works with Tsit5 and Bosh3
sol2 = solve(prob,alg, saveat=0.05)
usol2 = transpose(hcat(sol2.u...))
time = sol.t

plot(sol.t, usol2)
plot(sol.t, usol)
plt.xlabel("Time")
plt.show()

writedlm("test_basic_check_3.txt", [sol.t usol])
writedlm("test_basic_check_4.txt", [sol.t usol2])

# Basic check 5 & 6 =========================================
function basic_check_5(du, u, h, p, t)
    tau = p
    hist1 = h(p, t-tau)[1]
    du[1] =  u[1] * (1 - hist1)
end

h(p,t) = 1.2 * ones(1)   
u0 = [1.2]
tau = 4.0
lags = [tau]
p = (tau)
tspan = (0.0, 50.0)


prob = DDEProblem(basic_check_5, u0, h, tspan, p; constant_lags=lags)
alg = MethodOfSteps(BS3()) # doesn't work with DP5 DP8 but works with Tsit5 and Bosh3
sol = solve(prob,alg, saveat=0.05)
usol = transpose(hcat(sol.u...))

alg = MethodOfSteps(Tsit5()) # doesn't work with DP5 DP8 but works with Tsit5 and Bosh3
sol2 = solve(prob,alg, saveat=0.05)
usol2 = transpose(hcat(sol2.u...))
time = sol.t

plot(sol.t, usol2)
plot(sol.t, usol)
plt.xlabel("Time")
plt.show()

writedlm("test_basic_check_5.txt", [sol.t usol])
writedlm("test_basic_check_6.txt", [sol.t usol2])

# Basic check 7 =========================================
function basic_check_7(du, u, h, p, t)
    tau = p
    hist1 = h(p, t-tau)[1]
    du[1] =  u[1] * (1 - hist1)
end

h(p,t) = 1.2 * ones(1)   
u0 = [1.2]
tau = 4.0
lags = [tau]
p = (tau)
tspan = (0.0, 50.0)


prob = DDEProblem(basic_check_7, u0, h, tspan, p; constant_lags=lags)
alg = MethodOfSteps(Kvaerno5()) # doesn't work with DP5 DP8 but works with Tsit5 and Bosh3
sol = solve(prob,alg, saveat=0.05)
usol = transpose(hcat(sol.u...))
time = sol.t

plot(sol.t, usol)
plt.xlabel("Time")
plt.show()

writedlm("test_basic_check_7.txt", [sol.t usol])

# Basic check 8 =========================================
function basic_check_8(du, u, h, p, t)
    tau1, tau2 = p
    hist1 = h(p, t-tau1)[1]
    hist2 = h(p, t-tau2)[1]
    du[1] = - hist1 - hist2   
end

h(p,t) = 1.2 * ones(1)     
lags = [1.0/3  1.0/5]
p    = (1.0/3, 1.0/5)
tspan = (0.0, 10.0)
u0 = [1.2]

prob = DDEProblem(basic_check_8, u0, h, tspan, p; constant_lags=lags)
alg = MethodOfSteps(Tsit5())
sol = solve(prob,alg, saveat=0.1)
usol = transpose(hcat(sol.u...))
time = sol.t

plot(sol.t, usol)
plt.xlabel("Time")
plt.show()

writedlm("test_basic_check_8.txt", [sol.t usol])

# Basic check 9 =========================================
function basic_check_9(du, u, h, p, t)
    tau = p
    hist1 = h(p, t-tau)[1]
    du[1] =  0.2 * hist1 / (1+ hist1^10) - 0.1 * u[1] 
end


h(p,t) = 1.2 * ones(1)   
u0 = [1.2]
tau = 6.0
lags = [tau]
p = (tau)
tspan = (0.0, 50.0)

prob = DDEProblem(basic_check_9, u0, h, tspan, p; constant_lags=lags)
alg = MethodOfSteps(Tsit5())
sol = solve(prob,alg, saveat=0.1)
usol = transpose(hcat(sol.u...))
time = sol.t

plot(sol.t, usol)
plt.xlabel("Time")
plt.show()

writedlm("test_basic_check_9.txt", [sol.t usol])

# Basic check 10 =========================================
function basic_check_10(du, u, h, p, t)
    hist1 = h(p, t- 2 -sin(t))[1]
    du[1] =  u[1] * (1 - hist1)
end

function h_basic_check_10(p, t)
    1.2 * ones(1)  
end

prob = DDEProblem(basic_check_10,  h_basic_check_10,  (0.0, 40.0) ; dependent_lags = ((u, p, t) -> 2 + sin(t),))
alg = MethodOfSteps(BS3())
sol = solve(prob,alg, saveat=0.1)
usol = transpose(hcat(sol.u...))
time = sol.t

plot(sol.t, usol)
plt.xlabel("Time")
plt.show()

writedlm("test_basic_check_10.txt", [sol.t usol])

Basic check 11 =========================================
function basic_check_11(du, u, h, p, t)
    hist1 = h(p, t- 1/2*(exp(-u[1]^2) + 1))[1]
    du[1] =  -10*  hist1
end

function h_basic_check_11(p, t)
    1.0 * ones(1)  
end

prob = DDEProblem(basic_check_11,  h_basic_check_11,  (0.0, 5.0) ; dependent_lags = ((u, p, t) -> 1/2*(exp(-u[1]^2) + 1),))
alg = MethodOfSteps(Kvaerno5())
sol = solve(prob,alg, saveat=0.01)
usol = transpose(hcat(sol.u...))

alg = MethodOfSteps(Kvaerno4())
sol = solve(prob,alg, saveat=0.01)
usol2 = transpose(hcat(sol.u...))

time = sol.t

plot(sol.t, usol2, label="Kv4")
plot(sol.t, usol, label="Kv5")
plt.xlabel("Time")
plt.legend()
plt.show()

writedlm("test_basic_check_11.txt", [sol.t usol])


# Numerical check 1 =========================================
function numerical_check_1(du, u, h, p, t)
    hist1 = h(p, u[1])[1]
    du[1] =   hist1
end

function h_numerical_check_1(p, t)
    if t < 2.0
        1/2 * ones(1)  
    else 
        1 * ones(1)
    end
end

prob = DDEProblem(numerical_check_1,  h_numerical_check_1, u0=[1.0 * ones(1)], (2.0, 5.5) ; dependent_lags = ((u, p, t) -> u[1],))
alg = MethodOfSteps(BS3())
sol = solve(prob,alg, saveat=0.01)
usol = transpose(hcat(sol.u...))

alg = MethodOfSteps(Tsit5())
sol = solve(prob,alg, saveat=0.01)
usol2 = transpose(hcat(sol.u...))

time = sol.t

plot(sol.t, usol2, label="Kv4")
plot(sol.t, usol, label="Kv5")
plt.xlabel("Time")
plt.legend()
plt.show()

writedlm("test_basic_numerical_check_1.txt", [sol.t usol])

# Numerical check 2 =========================================
function numerical_check_2(du, u, h, p, t)
    hist1 = h(p, log(u[1]))[1]
    du[1] =   hist1 * u[1] / t 
end

function h_numerical_check_2(p, t)
    1 * ones(1)
end

prob = DDEProblem(numerical_check_2,  h_numerical_check_2, (1.0, 10.0) ; dependent_lags = ((u, p, t) -> log(u[1]),))
alg = MethodOfSteps(BS3())
sol = solve(prob,alg, saveat=0.01)
usol = transpose(hcat(sol.u...))
time = sol.t

plot(sol.t, usol2, label="Kv4")
plot(sol.t, usol, label="Kv5")
plt.xlabel("Time")
plt.legend()
plt.show()

writedlm("test_basic_numerical_check_2.txt", [sol.t usol])
