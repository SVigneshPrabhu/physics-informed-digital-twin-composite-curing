# Physics-Informed Digital Twin for Composite Curing

An AI-driven digital twin that uses reinforcement learning to autonomously optimize curing cycles for thick composite laminates used in aerospace manufacturing.

## Overview

Curing thick composite laminates is a slow, trial-heavy process — getting the temperature profile wrong leads to uneven cure, weak spots, or scrapped parts. This project replaces physical trial-and-error with a physics-informed RL agent that learns the optimal heating strategy in simulation, cutting both cost and development time.

A PPO (Proximal Policy Optimization) agent was trained to control multi-zone heating profiles, taking into account real physical constraints — tool thermal inertia, epoxy cure kinetics, and heat transfer — rather than treating the process as a black box.

## Key Results

- **AI policy achieves 95.0% cure completion**, safely approaching but not exceeding the 180°C thermal safety limit
- **Industry-standard baseline reaches only 43.2% cure** in the same time window — it fails to complete curing
- AI cycle time: 196.3 min (baseline shown is a partial/incomplete cure, so cycle times aren't directly comparable)
- Near-miss constraint penalties (ramp rate, thermal lag) were used during training to keep the policy within safe operating bounds, not just optimize for speed

## How It Works

1. **Physical model**: A CFD-coupled thermal model simulates heat transfer through the composite and tool, incorporating epoxy cure kinetics and tool thermal inertia.
2. **RL environment**: The simulation is wrapped as a custom environment where the agent controls multi-zone heater setpoints over time.
3. **Training**: A PPO agent (via TensorFlow) learns to adjust heating in real time to maximize cure uniformity while respecting thermal and process constraints.
4. **Evaluation**: Trained policies are benchmarked against standard fixed-profile curing to quantify uniformity and time savings.

## Tools & Technologies

- **Python** — core implementation
- **TensorFlow** — reinforcement learning (PPO agent)
- **CFD simulation coupling** — physical thermal/cure modeling

## Motivation

This was built as a personal research project to explore how physics-informed machine learning can accelerate manufacturing optimization in aerospace composites — an area where physical testing is expensive and slow, but full black-box ML risks missing physical constraints that matter for real parts.

## Status

Personal research project, 2025. Feedback and discussion welcome.
