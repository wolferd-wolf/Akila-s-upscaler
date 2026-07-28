# Akila's Upscaler

A temporal upscaler for mobile, built from scratch, inspired by the
algorithmic ideas in AMD FidelityFX Super Resolution 2 (FSR2). This is
an independent implementation targeting mobile GPU architectures
(tile-based deferred rendering — Mali, Adreno), not a fork of AMD's code.

## Why this is different from FSR2 desktop

FSR2's reference implementation assumes immediate-mode rendering with
cheap random framebuffer access. Mobile tile-based GPUs make history
buffer reprojection expensive if ported naively. This project adapts
the algorithm, not just the code, for that constraint.

## Structure

- `ref/` — Python reference implementation. Ground truth for correctness.
  Every shader must match this output within tolerance before being trusted.
- `shaders/` — GLSL compute shaders, ported from `ref/` once validated.
- `android/` — Vulkan app shell for on-device testing.
- `.github/workflows/` — CI: runs `ref/` tests on every push.

## Status

Phase 1: Python reference implementation (Prepare pass) — in progress.

## Build phases

1. Python reference: Prepare pass (motion vectors, depth-based reprojection validity)
2. Python reference: Resolve pass (history blend, clamping/rejection)
3. Port validated math to GLSL compute shaders
4. Minimal Android Vulkan harness (swapchain + compute pipeline)
5. Tile-GPU bandwidth optimization pass (subpass-friendly history access)

Phases 1–3 are CI-testable. Phase 4 onward requires manual on-device testing —
no GPU runners available in free CI tiers.
