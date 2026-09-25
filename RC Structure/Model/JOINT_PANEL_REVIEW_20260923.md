# Joint panel review, 2026-09-23

## Disposition

Keep `Joint_Panel.py` as a diagnostic prototype. Do not install its shared
beam-core topology or synthetic parameters in the building yet. IMKPinching
remains available for the user's requested joint-model research. This review
does not reverse that choice or change production settings.

Two issues need to be resolved separately: the physical role of degrading
joints in the intended SMRF population, and compatibility of the 3D macro
model. Matching a material loop cannot settle either issue.

## The perpendicular beam test

The implemented layout makes all four beam faces share one core rotation.
For shear-plane rotation q about global X, both X beams therefore twist by q.
With a fixed column core, elastic panel stiffness Kj, two transverse beam
torsional stiffnesses GJ/L, and applied moment M, our derivation is:

    Kt = sum(GJ/L)
    M = Kj*q + Kt*q
    q = M/(Kj + Kt)
    M_panel/M = 1/(1 + Kt/Kj)

The native OpenSees regression `test_perpendicular_beam_torsion_competes_with_panel_shear`
checks this equation using actual 3D elastic transverse beams. For synthetic
Kj = 100,000 kip-in/rad and M = 10 kip-in:

| Kt/Kj | q (rad) | Panel moment (kip-in) | Transverse torque (kip-in) |
|---:|---:|---:|---:|
| 0 | 0.0001000000 | 10.000000 | 0.000000 |
| 0.1 | 0.0000909091 | 9.090909 | 0.909091 |
| 1 | 0.0000500000 | 5.000000 | 5.000000 |
| 10 | 0.0000090909 | 0.909091 | 9.090909 |

This is a quantitative sensitivity result for the prototype, not proof that
all transverse torque is spurious. It shows that the layout has an important
structural assumption that isolated spring/rigid-arm checks do not validate.
Do not tune pinching parameters to compensate for unverified compatibility.

## Primary-source findings

NIST GCR 17-917-46v3, section 2.3, printed page 2-13 (PDF page 33), indicates
that joint yielding/deterioration need not be modeled for special moment
frames satisfying seismic joint confinement and anchorage requirements.
Finite joint size and bond-slip flexibility still matter. Appendix A.3
(printed A-3, PDF page 81) concerns nonductile joints and warns against
duplicating slip between joint and member models. This supports evaluating
a finite-joint, flexural-hinge baseline first; it is not evidence that every
randomly generated design satisfies all those assumptions, nor a prohibition
on researching joint degradation beyond that domain.

Source: [NIST report](https://doi.org/10.6028/NIST.GCR.17-917-46v3).

Altoontash (2004), sections 2.5.1-2.5.4, printed pp. 41-51, and Figures 2-27/28
(printed p. 72), uses six face nodes and a nine-DOF internal node separating
rigid-body motion from three shear modes. Face translation and rotation
constraints are defined separately. Our two rigid core bodies are not that
formulation. This source is a mechanics reference, not experimental
calibration of the current prototype.

Source: [original dissertation](https://opensees.berkeley.edu/OpenSees/doc/Altoontash_Dissertation.pdf).

The downloaded current `Joint3D.cpp` and `MP_Joint3D.cpp` were also inspected.
An isolated capability probe of installed OpenSees 3.8.0 attempted the documented
12-integer constructor. It failed with the warning that the constructor with
damage is not implemented. Thus the element is not an immediately usable
replacement in this installed build through that tested command. This is a
local capability result, not a claim that every OpenSees build lacks Joint3D.
Do not change OpenSees versions or implement a custom 3D constraint system
without a bounded compatibility plan.

Sources: [Joint3D](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/joint/Joint3D.cpp),
[MP_Joint3D](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/joint/MP_Joint3D.cpp).

## Experimental evidence screening

| Source | Evidence inspected | Applicability / next extraction |
|---|---|---|
| Kurose, Guimaraes, Liu, Kreger and Jirsa (1988), UT report 88-2 | University abstract; full scanned report retrieved | Three bidirectionally loaded joints with slabs. Candidate for 3D compatibility and response comparison. Specimen detailing, axial ratio, joint shear measurements and compliance with the intended design domain still require page-by-page extraction. No parameters accepted. |
| Leon and Jirsa (1986), Bidirectional Loading of R.C. Beam-Column Joints | Publisher abstract | Fourteen subassemblages; beam/slab geometry, bond and column-to-beam strength ratio matter. Useful candidate for loading-history and transverse-member effects. Full-text/specimen screening pending. |
| Golias and De Risi (2026), full-scale exterior joints | Modeling description reviewed previously | Low-standard exterior joints. Useful shear/slip separation example; unsuitable for transferring degrading parameters to SMRF joints without justification. |

Links: [UT study and report](https://utw10109.utweb.utexas.edu/research/publications/details-707094834/study-of-reinforced-concrete-beam-column-joints-under-uniaxial-and-biaxial-loading),
[Leon and Jirsa](https://doi.org/10.1193/1.1585397),
[Golias and De Risi](https://doi.org/10.3390/buildings16081638).

This is a screened shortlist, not a systematic review or a completed
experimental fit. Full source PDFs and extracted text remain in the local
analysis workspace; source URLs and byte hashes are included in the handoff.

## Next engineering work

1. Resolve the intended role of joint deterioration for verified SMRF designs:
   baseline behavior, an explicit sensitivity study, or an extended damage domain.
2. Extract a suitable 3D specimen's geometry, reinforcement, boundary conditions,
   axial history, applied loading, joint shear distortion, member-end rotations,
   and slab/transverse-member configuration. Preserve missing measurements.
3. Select and verify compatible 3D kinematics before fitting cyclic parameters.
   Compare physical face boundary conditions and work-conjugate variables;
   internal center-node restraints are not automatically equivalent across models.
4. Keep shear-only and combined shear/slip fits distinct. Existing member
   calibration includes `BOND_SLIP_INDICATOR = 1`; no switch to flexure-only
   behavior has been experimentally validated.
5. Reconcile finite joint faces with clear member lengths, loads, masses,
   diaphragms and member identities, then run one matched building diagnostic.

No new ground-motion run, data generation, production release, or specimen
calibration was performed in this review. The SSD is not required for the
tests or this report, and was not read during this review.
