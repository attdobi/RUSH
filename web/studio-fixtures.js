/* Deliberately illustrative: no measured quality, real labels or production lineage. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.RushStudioFixtures = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const node = (id, title, parent, kind, body) => ({id, title, parent, node_type: kind, body, status: 'illustrative'});
  const stages = [
    ['Define the policy', 'Start with intent, not a thousand exceptions.', 'A small, explicit policy establishes the label space. Golden labels are evidence, not instructions to memorize.', 'seed'],
    ['Find the boundary', 'Disagreement reveals where the rule is incomplete.', 'A new theme enters the review queue. Experts resolve the case and articulate what would change their decision.', 'evidence'],
    ['Learn a general rule', 'Turn an adjudication into reusable policy.', 'A bounded generalization is accepted in this scripted example. Keep its hard negative alongside it, separate evidence from policy, and preserve the parent intent.', 'accepted'],
    ['Reject the shortcut', 'A larger graph is not automatically a better policy.', 'The illustrative gate rejects an example-specific shortcut. The incumbent graph stays unchanged. Gates reduce risk; they do not prove generalization.', 'rejected'],
    ['Accept a bounded change', 'Promote a rule, with its boundary intact.', 'This walkthrough illustrates acceptance, not a measured win. Real promotion needs validation, slice checks and a protected final evaluation.', 'accepted'],
    ['Make the path explicit', 'Route known facts. Escalate uncertainty.', 'An explicitly authored shadow program demonstrates typed, version-pinned routing. It does not execute the live image judges.', 'shadow']
  ];
  function build(demo) {
    const mnist = demo === 'mnist', prefix = mnist ? 'MD' : 'GA';
    const area = mnist ? 'MNIST_Digits' : 'Generative_AI';
    const root = node(`${prefix}.root`, mnist ? 'Digit policy' : 'Image provenance', null, 'root', mnist ? 'Describe the structural features that distinguish digits. Ambiguous observations must remain reviewable.' : 'Assess supplied evidence of generation. Visual appearance alone is not proof of image origin.');
    const base = mnist ? [root,
      ...['0','1','2','3','4','5','6','7','8','9'].map(d => node(`MD.digit.${d}`, `Digit ${d}`, root.id, 'category', `Illustrative guideline for digit ${d}. Use structural features, not the identity of a training image.`))
    ] : [root,
      node('GA.provenance', 'Provenance evidence', root.id, 'category', 'Prefer a verified generation record over a visual guess. Missing metadata is not negative proof.'),
      node('GA.geometry', 'Scene geometry', root.id, 'category', 'Inconsistent geometry can motivate review, but is not a ground-truth origin label.'),
      node('GA.texture', 'Surface & texture', root.id, 'category', 'Repeated texture and unusual surfaces are supporting cues, not standalone proof.'),
      node('GA.exceptions', 'Hard negatives', root.id, 'exception', 'Real photographs, edits and compression artifacts can resemble synthetic imagery.'),
      node('GA.text', 'Text & symbols', root.id, 'category', 'Malformed symbols are review cues, not proof of generation.'),
      node('GA.anatomy', 'Anatomy & objects', root.id, 'category', 'Unusual anatomy may be a pose, an edit or a model artifact. Retain alternatives.')];
    const additions = mnist ? [
      node('MD.digit.4.boundary', '4 ↔ 9 boundary', 'MD.digit.4', 'boundary', 'A top closure alone is insufficient. Distinguish an angular crossbar from a rounded upper loop with a descending tail.'),
      node('MD.digit.9.tail', 'Loop + descending tail', 'MD.digit.9', 'rule', 'When both a rounded upper loop and a descending tail are reliably observed, propose 9. Otherwise preserve uncertainty.'),
      node('MD.digit.4.crossbar', 'Angular crossbar', 'MD.digit.4', 'rule', 'An angular crossbar supports a 4 interpretation. Do not identify an example by sample id, author or background.'),
      node('MD.digit.4.exception', 'Closed-top 4', 'MD.digit.4', 'exception', 'Some 4s have closed tops. Closure is not by itself a valid discriminator.'),
      node('MD.root.uncertain', 'Unknown → expert', root.id, 'boundary', 'Missing, contradictory or low-confidence observations go to expert review. Deterministic execution does not make visual extraction deterministic.'),
      node('MD.digit.1.flag', 'Flagged 1', 'MD.digit.1', 'exception', 'A small top flag on 1 need not imply 7.'),
      node('MD.digit.7.topbar', 'Bar + diagonal', 'MD.digit.7', 'rule', 'Compare the top bar and descending stroke jointly.'),
      node('MD.digit.3.open', 'Open left side', 'MD.digit.3', 'rule', 'Two right-facing bowls with an open left side support 3.'),
      node('MD.digit.5.stem', 'Top bar + stem', 'MD.digit.5', 'rule', 'The upper bar and left stem help distinguish 5 from 3.'),
      node('MD.digit.8.loops', 'Two closed loops', 'MD.digit.8', 'rule', 'Require reliable loop observations; pixel gaps can be ambiguous.'),
      node('MD.digit.6.loop', 'Lower loop', 'MD.digit.6', 'boundary', 'Loop position helps separate 6 from 9, but rotation must be considered.'),
      node('MD.digit.0.slash', 'Slashed zero', 'MD.digit.0', 'exception', 'A slash may be writing style rather than another digit.')
    ] : [
      node('GA.geometry.boundary', 'Stylized ≠ generated', 'GA.geometry', 'boundary', 'Stylization is a new review theme, not sufficient evidence of generation.'),
      node('GA.provenance.verified', 'Verified generation record', 'GA.provenance', 'rule', 'Only a record verified by a trusted upstream system may satisfy this predicate. A filename or model claim is not verification.'),
      node('GA.exceptions.compression', 'Compression artifacts', 'GA.exceptions', 'exception', 'Compression can introduce visual artifacts in real images. Preserve this hard negative.'),
      node('GA.texture.review', 'Visual cues → review', 'GA.texture', 'rule', 'Route suspicious visual cues for review; do not infer not-generated merely because a cue is absent.'),
      node('GA.provenance.unknown', 'Missing ≠ negative', 'GA.provenance', 'boundary', 'Absent or conflicting provenance remains unknown. Request evidence or expert review rather than defaulting to a negative label.'),
      node('GA.geometry.reflection', 'Reflections', 'GA.geometry', 'rule', 'Inconsistent reflections support review, not a conclusive origin judgment.'),
      node('GA.geometry.lighting', 'Light consistency', 'GA.geometry', 'rule', 'Consider occlusion and multiple light sources before calling a conflict.'),
      node('GA.anatomy.hands', 'Hands & occlusion', 'GA.anatomy', 'boundary', 'Occlusion and motion may explain apparent anatomical anomalies.'),
      node('GA.anatomy.objects', 'Object continuity', 'GA.anatomy', 'rule', 'Check continuity across visible object parts and preserve ambiguity.'),
      node('GA.text.glyphs', 'Glyph consistency', 'GA.text', 'rule', 'Unreadable text can arise from distance or compression; ask for corroboration.'),
      node('GA.text.logos', 'Logo distortion', 'GA.text', 'boundary', 'Distortion is a weak cue without reliable reference evidence.'),
      node('GA.texture.repetition', 'Repeated patterns', 'GA.texture', 'rule', 'Repetition can be natural, manufactured or synthetic; do not decide from it alone.'),
      node('GA.exceptions.art', 'Traditional artwork', 'GA.exceptions', 'exception', 'Human artwork may be stylized without being generated.'),
      node('GA.provenance.chain', 'Evidence chain', 'GA.provenance', 'rule', 'Verify provenance with a trusted upstream process; source tags alone are not signatures.'),
      node('GA.exceptions.edit', 'Ordinary image edits', 'GA.exceptions', 'exception', 'Separate ordinary editing from the policy definition of generated imagery.')
    ];
    const counts = mnist ? [0,2,5,5,9,12] : [0,3,6,6,11,15];
    const frames = stages.map(([title, headline, detail, status], i) => {
      const nodes = [...base, ...additions.slice(0, counts[i])].map(n => ({...n}));
      if (i >= 4) nodes.find(n => n.id === additions[0].id).body += ' Retain the exception alongside the positive rule.';
      const edges = nodes.filter(n => n.parent).map(n => ({source:n.id, target:n.parent, type:'subtype_of'}));
      if (i >= 2) edges.push({source:additions[0].id, target:additions[1].id, type:'distinguishes'});
      if (i >= 4) edges.push({source:additions[3].id, target:additions[1].id, type:'constrains'});
      return {version:`demo-v${i}`, title, headline, detail, status, area, nodes, edges, origin:'illustrative'};
    });
    const rule = (id, title, policy, field, yes, no) => ({id, title, kind:'rule', policy_node_id:policy, when:{field, op:'eq', value:true, sources:['sme','system'], min_confidence:0.9}, next:{true:yes, false:no, unknown:'review'}});
    const program = {schema_version:1, id:`${demo}-shadow-example-v1`, mode:'shadow', policy_area:area, policy_version:'demo-v5', entry:'first', nodes:mnist ? [
      rule('first','Rounded upper loop?', 'MD.digit.9', 'rounded_loop','second','third'),
      rule('second','Descending tail?', 'MD.digit.9.tail','descending_tail','nine','review'),
      rule('third','Angular crossbar?', 'MD.digit.4.crossbar','angular_crossbar','four','review'),
      {id:'nine',kind:'action',title:'Suggest digit 9',action:'suggest_digit_9'},
      {id:'four',kind:'action',title:'Suggest digit 4',action:'suggest_digit_4'},
      {id:'review',kind:'action',title:'Expert review',action:'sme_review'}
    ] : [
      rule('first','Conflicting provenance?', 'GA.provenance.unknown','conflicting_provenance','review','second'),
      rule('second','Verified generation?', 'GA.provenance.verified','verified_generation_record','generated','third'),
      rule('third','Suspicious visual cues?', 'GA.texture.review','visual_risk','review','evidence'),
      {id:'generated',kind:'action',title:'Suggest generated',action:'suggest_generated'},
      {id:'evidence',kind:'action',title:'Request provenance',action:'request_provenance'},
      {id:'review',kind:'action',title:'Expert review',action:'sme_review'}
    ]};
    const fact = value => ({value,source:'sme',confidence:1});
    const scenarios = mnist ? [
      {title:'Clear 9',facts:{rounded_loop:fact(true),descending_tail:fact(true)}},
      {title:'Angular 4',facts:{rounded_loop:fact(false),angular_crossbar:fact(true)}},
      {title:'Missing evidence',facts:{}},
      {title:'Low confidence',facts:{rounded_loop:{value:true,source:'model',confidence:0.5}}}
    ] : [
      {title:'Verified record',facts:{conflicting_provenance:fact(false),verified_generation_record:fact(true)}},
      {title:'Conflicting evidence',facts:{conflicting_provenance:fact(true)}},
      {title:'Missing evidence',facts:{}},
      {title:'No visual warning',facts:{conflicting_provenance:fact(false),verified_generation_record:fact(false),visual_risk:fact(false)}}
    ];
    return {area,frames,program,scenarios};
  }
  return {build};
});
