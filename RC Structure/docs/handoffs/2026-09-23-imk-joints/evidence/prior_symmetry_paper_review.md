# IMK calibration-source review: Symmetry 18 (2026), 1493

Reviewed 2026-09-22 for the 3D RC SMRF model in StructGNN-OpenSeesPy-Integrated.

**Disposition: retain as a calibration-method reference and an optional pier reproduction example. Do not adopt its numerical parameters for our beam, column, or joint springs.** This is a source assessment, not a rejection of the paper's entire method. No production parameters or release decisions were changed.

## Source and scope

Duan et al., *Calibration and Evaluation of an IMK Hysteretic Model for Seismic Fragility Assessment of Urban Rail RC Solid Piers*, Symmetry 2026, 18, 1493. DOI: https://doi.org/10.3390/sym18091493.

Source file: `C:/Users/andro/Downloads/ResearchProject/symmetry-18-01493.pdf`; 43 pages; 20,122,828 bytes. SHA-256: `6a1af654c9fd97c77a4748933955a70c68ddcc89760d32f704e71a3e9a76ae9e`.

The paper reuses seven quarter-scale RC cantilever-pier tests. All have a reported axial-load ratio of 0.05. The 500 mm square specimens vary height, longitudinal reinforcement and transverse reinforcement. Heights are 1.95, 2.95 and 3.95 m; the material and reinforcement details appear in Tables 1 and 2, page 11. These are pier tests, not beam-column joint tests or validation of a complete 3D SMRF. The original experimental source is reference 47: Duan et al., *Seismic Fragility of Urban Rail Transport RC Solid Piers Considering Multiparameter Effects*, Buildings 2026, 16, 2327. That original source has not been independently reviewed here.

## Useful method

Sections 2.2 and 3.2 establish positive and negative backbones, identify yield and peak points, fit the descending branch, and then adjust cyclic parameters against complete loops. The paper separates backbone fitting from unloading degradation, pinching and energy-deterioration calibration. This is a useful sequence for our component fixtures.

For our research, compare at least backbone force, unloading stiffness, repeated-cycle strength loss, loop energy and residual deformation. Preserve a specimen or loading history for an independent check when the available data permit it. Record the source specimen, geometry, axial loading, units, deformation definition, fitted quantities and applicability range alongside each parameter set.

## What cannot be transferred directly

### Whole-pier response versus separate mechanisms

Page 15 maps global pier force and displacement into one base spring through M = F L, theta = Delta/L and K_theta = k_e L^2. This equivalent global response does not separately identify member flexure, joint shear and anchorage slip. It therefore cannot directly calibrate our proposed separate IMKPeakOriented member-end springs and IMKPinching joint springs. The paper's pinching response is not experimental validation of a beam-column joint material assignment.

For our model, the deformation budget must account for elastic members, member plastic rotations and joint/anchorage deformation consistently. Existing member calibration already includes a bond-slip contribution; adding a joint spring without reconciling that contribution risks representing the same deformation twice. Merely switching a material name does not resolve that issue.

### Energy normalization needs clarification

Equation (5), page 4, defines reference energy as E_t = gamma F_y Delta_y. Page 20 later reports capital-Lambda values: S = C = 500, A = 271, K = 72, without explicitly establishing their relationship to gamma or their deformation coordinate. The paper text does not identify an exact `IMKPinching` command or provide a runnable material declaration.

Current official OpenSees documentation for both IMKPinching and IMKPeakOriented defines E_ref = Lamda F_y, where F_y is the material's generalized yield force (moment for a rotational spring). Consequently, gamma in Equation (5) cannot automatically be copied into a modern Lamda argument.

Conditional derivation using the paper's equivalent global cantilever coordinate:

    M_y = F_y L
    theta_y = Delta_y / L
    gamma F_y Delta_y = gamma M_y theta_y
    Lamda_OpenSees = gamma theta_y

For the page-20 prototype, theta_y = 120 / 12000 = 0.010 rad. IF the reported values 500, 271 and 72 denote gamma, their corresponding rotational Lamda values would be 5.00, 2.71 and 0.72. IF those values already denote command-level Lamda, applying this conversion again would be incorrect. **These are conditional calculations, not adopted inputs or a claim that the authors used one interpretation.** The relationship must be resolved from their implementation before numerical reuse.

### Elastic compliance and geometric nonlinearity

Page 15 describes both the global-to-spring conversion and an elastic pier element. For a simple elastic cantilever with a base rotational spring, tip-load compliance is:

    1/k_global = L^3/(3 E I) + L^2/K_spring

Thus K_spring = k_global L^2 accounts for the entire global compliance by itself. A separate finite-flexibility shaft requires a documented stiffness partition. This is a reproduction question arising from the stated equations; it is not proof of an error in the unavailable author implementation.

Page 20 explicitly states that the prototype's fitted backbone already contains P-Delta effects from a fiber model under constant 12.8 MN axial load. Its IMK model uses a linear transformation to avoid adding those effects again. Copying that backbone into our explicitly geometrically nonlinear 3D frame would require separating the geometric contribution. The paper itself limits this fixed-envelope treatment under variable axial force and bidirectional interaction.

## Specific reproducibility and evidence limits

1. **Repeated table rows.** Table 3, page 14, reports exactly the same negative-direction backbone entries for A1, A2 and A3: Delta_y = 35.09 mm, F_y = 319.31 kN, k_e = 9.10 kN/mm, Delta_pk = 65.03 mm, F_pk = 357.29 kN, U_pre = 0.85, U_post = 2.93, U_u = 4.79, k_py = 1.12. This repetition was confirmed visually. The specimens have different heights, and the plotted responses differ. Seek clarification before reproducing those rows; do not silently repair them or infer misconduct.
2. **One elastic stiffness in the current command.** Tables 3 and 5 list separate positive and negative elastic stiffnesses, while modern IMKPinching and IMKPeakOriented each accept one Ke. Reproduction requires an explicit mapping or explanation of a different implementation.
3. **Ultimate deformation is extrapolated.** Tests ended at approximately 80% of peak resistance (page 12). Equations (30)-(31), page 13, use a descending-line zero-force intercept. The resulting U_u is not a directly observed zero-strength collapse deformation. U_pre and U_post are normalized displacement quantities, not spring rotations in radians.
4. **Calibration is not independent validation.** There is no specimen holdout. The prototype backbone was extracted from the same fiber model subsequently used for comparisons. The authors explicitly acknowledge this (pages 14-15, 20, 39-40).
5. **Hysteretic energy remains a material limitation.** Across the seven calibration specimens, IMK cumulative-energy MAPE is 27.79%, with increasing underestimation at larger energy demand. Final secant-stiffness MAPE is 9.93%; residual-displacement R-squared is 0.486 (pages 17-18). Matching peak force or stiffness alone is insufficient evidence for the loop behavior we intend to generate.

## Consequence for our implementation

Proceed with configurable, explicitly identified calibration sets and component-level cyclic fixtures. Keep numerical calibration of member flexure and joint shear/slip grounded in sources that match those mechanisms and the relevant axial-load, detailing and loading domains. Treat this paper as a workflow example. It does not settle our joint topology, validate 3D coupling, justify using its constants across randomly generated buildings, or close the existing production qualification decisions.

The most productive next literature target is a cyclic beam-column subassembly dataset with reported joint shear or anchorage deformation, so the joint contribution can be distinguished from member flexure. Reproducing one pier from this paper is optional and should remain isolated from production.

## Primary implementation references checked

- https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/IMKPinching.html
- https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/IMKPeakOriented.html

These were checked on 2026-09-22. A final parameter adapter must also be checked against the installed OpenSees version. The PDF's equations, Table 3, Figure 9 and Table 5 were inspected as rendered pages in addition to text extraction. No author code or raw experimental histories were available in the supplied PDF; no author contact was made.
