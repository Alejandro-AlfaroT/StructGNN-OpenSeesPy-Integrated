from MP_TESTING import Test_SAP2000
from MP_TESTING import Del_Files
from MP_TESTING import graph_generation

# Units: lb_in_F = 1, lb_ft_F = 2, kip_in_F = 3, kip_ft_F = 4, kN_mm_C = 5, kN_m_C = 6, kgf_mm_C = 7, kgf_m_C = 8, N_mm_C = 9, N_m_C = 10, Ton_mm_C = 11, Ton_m_C = 12, kN_cm_C = 13, kgf_cm_C = 14, N_cm_C = 15, Ton_cm_C = 16
Units = 4

# 0 = OpenFrame, 1 = PerimeterFrame, 2 = BeamSlab, 3 = FlatPlate
template_type = 0
NumStory = 2
StoryHeight = 12
SpansX = 4
LengthX = 20
SpansY = 4
LengthY = 20

# 1 = Dead, 2 = SuperDead, 3 = Live, 4 = Reduced Live, 5 = Quake, 6 = Wind, 7 = Snow, 8 = Other... Check API for more
LType = 8

#Top Loads
ForceX = 90
ForceY = 150

#Set up PyTorch Graph
#In pytorch object Y is Z, and Z is Y compared to SAP2000 coordinate system
graph_generation.generate_structure(1, NumStory, SpansX, SpansY, LengthX, LengthY, StoryHeight, ForceX, ForceY)

#Run Analysis and save results as CSV file
Test_SAP2000.SAPAnalysis(Units, template_type, NumStory, StoryHeight, SpansX, LengthX, SpansY, LengthY, LType, ForceX, ForceY)

#Update output results in PyTorch Graph


#Delete the extra files made
Del_Files.cleanup_SAP2000()
#Del_Files.cleanup_csv()