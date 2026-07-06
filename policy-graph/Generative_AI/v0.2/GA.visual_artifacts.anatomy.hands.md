---
id: GA.visual_artifacts.anatomy.hands
version: Generative_AI.v0.1
title: Anatomical hand, limb, paw, and digit artifacts
area: Generative_AI
node_type: category
parent: GA.root
polarity: positive
status: draft
coverage_weight: 1.4
coverage_target:
  easy_positive: 30
  hard_positive: 30
  easy_negative: 15
  hard_negative: 30
  platinum_min: 8
source_anchors:
  - RUSH v0.1 worked GenAI example - hand/finger artifacts
  - RUSH holdout review 2026-05-11 - animal paw and claw artifacts missed
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.boundary.photo_editing}
  - {type: boundary_with, to: GA.boundary.low_quality_uncertain}
  - {type: boundary_with, to: GA.exception.medical_prosthetic}
canonical_examples: []
---
# Anatomical hand, limb, paw, and digit artifacts

## Positive criteria
Use this node when visible human or animal anatomy includes strong generative artifacts such as:

1. Extra, missing, fused, webbed, duplicated, or malformed fingers, toes, claws, hooves, paws, or digits without plausible real-world explanation.
2. Hands, feet, paws, legs, arms, tails, ears, or limbs that violate skeletal articulation, species anatomy, pose, or perspective.
3. Blended fingers or toes, palms without coherent thumbs, paws with incoherent pads or claws, or inconsistent limb continuity.
4. Animal forelegs or paws that merge into rocks, grass, fur, water, or shadow while the rest of the subject is sharp.
5. Teeth, muzzle, ear, horn, antler, fin, or tail structures that duplicate, melt, or connect impossibly to nearby anatomy.

## Hard negatives
- Motion blur, water spray, grass, shadow, fur, clothing, gloves, or occlusion that only appears like extra digits.
- Real medical conditions, injuries, prosthetics, amputations, species variation, grooming, or unusual but possible animal poses.
- Tiny crops where the relevant anatomy is not clearly visible.

## Boundary with medical and real-world variation
Do not classify atypical anatomy as `gen_ai` when it is plausibly explained by `[[GA.exception.medical_prosthetic]]`, species anatomy, injury, assistive devices, pose, or occlusion. Require an actual structural impossibility or a cluster of local synthesis cues.

## SME review triggers
Escalate when the hand, paw, limb, or digit region is cropped, highly stylized, medically atypical, species-ambiguous, or the artifact could plausibly come from editing/compression.
