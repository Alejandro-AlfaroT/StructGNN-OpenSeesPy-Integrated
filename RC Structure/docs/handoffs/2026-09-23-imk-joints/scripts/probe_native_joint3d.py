"""Read-only capability/kinematics probe of installed OpenSees, in a fresh process."""
import json
import openseespy.opensees as ops

ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)
coordinates = [(-12.,0.,0.), (12.,0.,0.), (0.,-15.,0.), (0.,15.,0.), (0.,0.,-10.), (0.,0.,10.)]
for tag, coordinates_i in enumerate(coordinates,1):
    ops.node(tag,*coordinates_i)
for tag, stiffness in ((11,1e5),(12,1.7e5),(13,1e5)):
    ops.uniaxialMaterial("Elastic",tag,stiffness)
report={"opensees_version":ops.version(),"status":"not_probed"}
try:
    ops.element("Joint3D",21,1,2,3,4,5,6,100,11,12,13,0)
    report.update(status="constructed",element_nodes=ops.eleNodes(21),center_dofs=ops.getNDF(100))
    ops.fix(100,1,1,1,1,1,1,0,0,1)
    ops.timeSeries("Linear",1)
    ops.pattern("Plain",1,1)
    ops.load(4,0.,0.,0.,10.,0.,0.)
    ops.constraints("Lagrange",1e5,1e5)
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.test("NormDispIncr",1e-10,50)
    ops.algorithm("Newton")
    ops.integrator("LoadControl",1.)
    ops.analysis("Static")
    code=ops.analyze(1)
    report.update(analyze_code=code,center_disp=ops.nodeDisp(100),face_disp={str(n):ops.nodeDisp(n) for n in range(1,7)})
except Exception as error:
    report.update(status="unavailable_or_failed",error=str(error))
print(json.dumps(report,indent=2))
ops.wipe()
