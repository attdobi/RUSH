---
id: GA.exception.medical_prosthetic
version: Generative_AI.v0.1
title: Medical conditions and prosthetics
area: Generative_AI
node_type: exception
parent: GA.visual_artifacts.anatomy.hands
polarity: hard-negative
status: draft
coverage_weight: 1.5
coverage_target:
  easy_negative: 20
  hard_negative: 30
  platinum_min: 10
source_anchors:
  - "v0.1 §4.1: polydactyly, syndactyly as hard negatives for hand artifacts"
  - "v0.2 §3: medical-conditions as exception_to anatomical nodes"
canonical_examples: []
---
# Medical conditions and prosthetics

## Decision rule
Do not classify anatomy as `gen_ai` when atypical hands, limbs, skin, or body structure are plausibly explained by real medical variation, assistive devices, prosthetics, injury, or treatment history.

## Positive criteria
Use this hard-negative exception for real-world conditions and devices that can be confused with `[[GA.visual_artifacts.anatomy.hands]]`, including:

1. Polydactyly, syndactyly, ectrodactyly, limb differences, amputations, congenital variations, or atypical digit count.
2. Prosthetic hands, arms, legs, fingers, sockets, braces, orthotics, adaptive grips, or visible assistive hardware.
3. Post-surgical reconstruction, scarring, grafts, trauma, swelling, burns, or healing wounds.
4. Dermatological conditions, vitiligo, psoriasis, severe dryness, makeup coverage, medical tape, or compression garments that can resemble `[[GA.surface_texture.plastic_skin]]`.

## Boundary with hand artifacts
Route to `[[GA.visual_artifacts.anatomy.hands]]` only when the anatomy violates plausible real-world structure without medical, device, pose, occlusion, or perspective explanation. Atypical anatomy is not itself evidence of generation.

## Boundary with plastic-skin artifacts
Route to `[[GA.surface_texture.plastic_skin]]` only when synthetic texture cues persist beyond explainable skin condition, prosthetic material, medical dressing, makeup, lighting, or compression.

## SME review triggers
Escalate rather than guess when medical status is unknown, when a prosthetic or brace is partly cropped, or when a real condition and a possible generative artifact overlap.
