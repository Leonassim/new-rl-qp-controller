mc_rtc new RL-QP controller template
==

This project is a template for a new RL-QP controller project wihtin [mc_rtc].

The goal of this template is to help users deploy reinforcement learning (RL) policies within a Quadratic Programming (QP) framework augmented with Control Barrier Functions (CBFs). This combination is enforcing physical and safety constraints, including:

* Joint position limits
* Joint velocity limits
* Torque limits
* Self-collision avoidance

Further details are available in:

[*Safe Execution of RL Policies via Acceleration-based CBF-QP Constraint Enforcement for Real-World Robotic Deployments*](https://hal.science/hal-05362571)


It comes with:
- a CMake project that can build a controller in [mc_rtc], the project can be put within [mc_rtc] source-tree for easier updates
- clang-format files
- automated GitHub Actions builds on three major platforms

Quick start
--

1. Renaming the controller from `NewRLQPController` to `MyController`. In a shell (Git Bash on Windows, replace sed with gsed on macOS):

```bash
sed -i -e's/NewRLQPController/MyController/g' `find . -not -path '*/.*' -type f`
git mv src/NewRLQPController.cpp src/MyController.cpp
git mv src/NewRLQPController.h src/MyController.h
git mv src/states/NewRLQPController_Initial.cpp src/states/MyController_Initial.cpp
git mv src/states/NewRLQPController_Initial.h src/states/MyController_Initial.h
git mv etc/NewRLQPController.in.yaml etc/MyController.in.yaml
```

2. You can customize the project name in vcpkg.json as well, note that this must follow [vcpkg manifest rules](https://github.com/microsoft/vcpkg/blob/master/docs/users/manifests.md)

2. Build and install the project

3. Run using your [mc_rtc] interface of choice, and setting `Enabled` to `MyController`

[mc_rtc]: https://jrl-umi3218.github.io/mc_rtc/
