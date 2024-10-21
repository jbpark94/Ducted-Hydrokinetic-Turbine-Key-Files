#!/usr/bin/env python
import os
import argparse
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
from pygeo import geo_utils, pyGeo
from stl import mesh

parser = argparse.ArgumentParser()
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================
# Case Parameters
U0 = -1.7#1.4
p0 = 0.0
D = 1.2901859*2
Drotor = 1.0*2
L = 1.866
A0 = 3.141592/4*D*D
angVel = -8.262#-10.694#-9.697 
#Urel = (U0**2 + (angVel*0.3)**2 )**0.5
#Lchord = 0.03124

Urel = 1.7
Lchord = 0.125

k0 = 3/2*(0.10*0.10)*(Urel*Urel)  
omega0 = k0**0.5/0.09**0.25/Lchord 

daOptions = {
    "designSurfaces": ["Duct","Blade"],
    "solverName": "DASimpleFoam",
    "useAD": {"mode": "reverse"},
    "primalMinResTol": 1.0e-9,
    "primalMinResTolDiff": 1.0e6,
    "objFunc": {
        "Torque": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["Blade"],
                "axis": [1.0, 0.0, 0.0],
                "center": [0.0, 0.0, 0.0],
                "scale": 1.0 * 999.1 * angVel / (0.5 * 999.1 * U0 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-10,
        "pcFillLevel": 2,
        "jacMatReOrdering": "natural",#"rcm",#"natural",
        "gmresMaxIters": 2500,
        "gmresRestart": 2500,
        "adjPCLag": 1,
    },
	"normalizeStates": {"U": 1, "p": U0*U0/2, "phi": 1, "nuTilda": 3e-6, "k": k0, "omega": omega0, "nut": 1}, 
    "checkMeshThreshold": {
        "maxAspectRatio": 2000.0,
        "maxNonOrth": 90.0,
        "maxSkewness": 6.0,
    },
    "designVar": {
        
        "duct_scale": {"designVarType": "FFD"},
        "ductAOA": {"designVarType": "FFD"},
        "cst_uDuct": {"designVarType": "FFD"},
        "cst_lDuct": {"designVarType": "FFD"},

        "chord": {"designVarType": "FFD"},
        "twist": {"designVarType": "FFD"},
        "Rblade": {"designVarType": "FFD"},
        "hub0_x": {"designVarType": "FFD"},
        "hub1_z": {"designVarType": "FFD"},
        "hub2_z": {"designVarType": "FFD"},
        "hub3_x": {"designVarType": "FFD"},

    },
    "decomposeParDict": {},
#    "writeMinorIterations":True
}

# Mesh deformation setup
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    "useRotations": True,
    "LdefFact": 1000.0,#100.0,
    "errTol": 1e-5,
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}

class ESPHack(OM_DVGEOCOMP):
    def setup(self):
        super().setup()
        self.it = 0

    def compute(self, inputs, outputs):
        super().compute(inputs, outputs)
        self.nom_getDVGeo().writeCSMFile(f"updated_{self.it:05d}.csm")
        # self.nom_getDVGeo().writeCADFile(f"updated_{self.it:05d}.stl")
        self.it += 1

# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):
        # =====================================================================
        # DAFoam Setup
        # =====================================================================
        dafoam_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        # =====================================================================
        # MPhys Setup
        # =====================================================================
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # Add mesh component
        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        # Add geometry component (FFD)
        self.add_subsystem("geometry", ESPHack(file="ESP/e423_5kW_2.csm", type="esp", options={"exclude_edge_projections": True, "projTol":0.2}))

        # Add scenario
        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

        #self.add_subsystem("first_cst", om.ExecComp("cst_con = cst_u + cst_l"))
 #       self.add_subsystem("first_cst", om.ExecComp("cst_con = (cst_u**2) / (cst_l**2)"))
 #       self.add_subsystem("first_cst_2", om.ExecComp("cst_con2 = cst_u - cst_l"))


    def configure(self):
        super().configure()

        self.cruise.aero_post.mphys_add_funcs()

        # Get surfaces from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # Add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # Create constraint DV setup
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)
        
        self.geometry.nom_addESPVariable("duct_scale", scale=1.5, dh=1e-6)
        self.geometry.nom_addESPVariable("ductAOA", scale=0.5, dh=1e-6)
        self.geometry.nom_addESPVariable("Rblade", scale=4.0, dh=1e-6)

        self.geometry.nom_addESPVariable("chord", scale=2.5, dh=1e-6)
        self.geometry.nom_addESPVariable("twist", scale=0.05, dh=1e-6)
        self.geometry.nom_addESPVariable("cst_uDuct", scale=1.5, dh=1e-6)
        self.geometry.nom_addESPVariable("cst_lDuct", scale=1.5, dh=1e-6)

        self.geometry.nom_addESPVariable("hub0_x", scale=25.0, dh=1e-6)
#        self.geometry.nom_addESPVariable("hub1_x", scale=25.0, dh=1e-6)
        self.geometry.nom_addESPVariable("hub1_z", scale=12.5, dh=1e-6)
#        self.geometry.nom_addESPVariable("hub2_x", scale=25.0, dh=1e-6)
        self.geometry.nom_addESPVariable("hub2_z", scale=12.5, dh=1e-6)
        self.geometry.nom_addESPVariable("hub3_x", scale=25.0, dh=1e-6)
        
        xdv = self.geometry.nom_getDVGeo().getValues()
        self.dvs.add_output("duct_scale", val=xdv["duct_scale"])
        self.dvs.add_output("ductAOA", val=xdv["ductAOA"])
        self.dvs.add_output("Rblade", val=xdv["Rblade"])

        self.dvs.add_output("chord", val=xdv["chord"])
        self.dvs.add_output("twist", val=xdv["twist"])
        self.dvs.add_output("cst_uDuct", val=xdv["cst_uDuct"])
        self.dvs.add_output("cst_lDuct", val=xdv["cst_lDuct"])

        self.dvs.add_output("hub0_x", val=xdv["hub0_x"])
#        self.dvs.add_output("hub1_x", val=xdv["hub1_x"])
        self.dvs.add_output("hub1_z", val=xdv["hub1_z"])
#        self.dvs.add_output("hub2_x", val=xdv["hub2_x"])
        self.dvs.add_output("hub2_z", val=xdv["hub2_z"])
        self.dvs.add_output("hub3_x", val=xdv["hub3_x"])

       
# run 1
        ptList = [[0.784730825,0,1.158408752],[0.663649032,0,1.146660457],[0.54264916,0,1.136974282],[0.421717821,0,1.129013173],[0.300856686,0,1.122819238],[0.1800656,0,1.118388549],[0.059342747,0,1.115675398],[-0.061314405,0,1.114616047],[-0.181908525,0,1.11514334],[-0.302442199,0,1.117192148],[-0.422917965,0,1.12069861],[-0.543338475,0,1.125595966],[-0.663706708,0,1.131809205],[-0.784026159,0,1.139250384],[-0.904300936,0,1.14781608],[-1.024535729,0,1.157388244],[-1.144735597,0,1.167839541],[-1.264905539,0,1.179044118],[-1.385049818,0,1.190894664],[-1.505171004,0,1.20332652],[-1.625268703,0,1.216349561],[-1.745337962,0,1.230088493],[-1.865367305,0,1.244832195],[-1.985336391,0,1.261092667],[-2.10521327,0,1.279674147]]

        self.geometry.nom_addThicknessConstraints1D("duct_thickcon",ptList,nCon=50,axis=[0,0,1],scaled=False)

        ptList2 = np.array([[-0.01,-0.017,1.0263]])
        vecList2 = np.array([[0.0,0.0,1.0]])
        surf_blade = mesh.Mesh.from_file("ESP/e423_5kW_blade_optTwist.stl")
        surf_duct = mesh.Mesh.from_file("ESP/e423_5kW_duct.stl")
        surf_blade_list = [surf_blade.v0, surf_blade.v1-surf_blade.v0, surf_blade.v2-surf_blade.v0]
        surf_duct_list = [surf_duct.v0, surf_duct.v1-surf_duct.v0, surf_duct.v2-surf_duct.v0]

        self.geometry.nom_setConstraintSurface(surface=surf_blade_list,name="surf_blade")
        self.geometry.nom_setConstraintSurface(surface=surf_duct_list,name="surf_duct")
        self.geometry.nom_addProximityConstraints("gap_con",ptList2,vecList2,surfA="surf_duct",surfB="surf_blade")

        self.connect("duct_scale", "geometry.duct_scale")
        self.connect("ductAOA", "geometry.ductAOA")
        self.connect("Rblade", "geometry.Rblade")

        self.connect("chord", "geometry.chord")
        self.connect("twist", "geometry.twist")
        self.connect("cst_uDuct", "geometry.cst_uDuct")
        self.connect("cst_lDuct", "geometry.cst_lDuct")

        self.connect("hub0_x", "geometry.hub0_x")
        self.connect("hub1_z", "geometry.hub1_z")
        self.connect("hub2_z", "geometry.hub2_z")
        self.connect("hub3_x", "geometry.hub3_x")

#        self.connect("cst_uDuct", "first_cst.cst_u", src_indices=0)
#        self.connect("cst_lDuct", "first_cst.cst_l", src_indices=0)
#        self.connect("cst_uDuct", "first_cst_2.cst_u", src_indices=0)
#        self.connect("cst_lDuct", "first_cst_2.cst_l", src_indices=0)
        

        # define the design variables to the top level
        self.add_design_var("duct_scale", lower=0.7, upper=1.3, scaler=1.5)
        self.add_design_var("ductAOA", lower=0.3, upper=10, scaler=0.5)
        Rblade_0 = 1.0
        self.add_design_var("Rblade", lower=0.98*Rblade_0, upper=1.02*Rblade_0, scaler=4.0)
        

        chord_0 = np.array([0.1204, 0.1157, 0.1063, 0.097, 0.0876, 0.0782, 0.0688, 0.0595, 0.05])
        chord_0_lb = chord_0*0.8
        chord_0_ub = chord_0*1.2
        chord_0_lb[0] = 0.1204*0.98
        chord_0_ub[0] = 0.1204*1.02

        twist_0 = np.array([27.3,  26.9,  25.1,  21.3,  17.2,  14.2,  12.7,  12.6,  13.1])
        twist_0_lb = twist_0*0.8#np.array([27.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        twist_0_ub = twist_0*1.2#np.array([27.3, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        twist_0_lb[0] = 27.3*0.98
        twist_0_ub[0] = 27.3*0.98

        self.add_design_var("chord", lower=chord_0_lb, upper=chord_0_ub, scaler=2.5)
        self.add_design_var("twist", lower=twist_0_lb, upper=twist_0_ub, scaler=0.05)
        cst_uDuct_0_lb = -0.8*np.ones(6)
        cst_lDuct_0_ub = 0.8*np.ones(6)
        cst_uDuct_0_lb[0] = 0.0316622777 # curvature radius 0.0005
        cst_lDuct_0_ub[0] = -0.0316622777 # curvature raidus 0.0005

        self.add_design_var("cst_uDuct", lower=cst_uDuct_0_lb, upper=0.8*np.ones(6), scaler=1.5)
        self.add_design_var("cst_lDuct", lower=-0.8*np.ones(6), upper=cst_lDuct_0_ub, scaler=1.5)

        Lhub1_0 = 0.73
        hub0_x0 = 0.25
        hub1_x0 = 0.2232
        hub1_z0 = 0.1
        hub2_x0 = 0.99325
        hub2_z0 = 0.18
        hub3_x0 = 1.2565

        self.add_design_var("hub0_x", lower=0.1, upper=0.5, scaler=1.0)
        self.add_design_var("hub1_z", lower=0.1, upper=0.2, scaler=1.0)
        self.add_design_var("hub2_z", lower=0.1, upper=0.2, scaler=1.0)
        self.add_design_var("hub3_x", lower=1.0, upper=1.51, scaler=1.0)

        # # add objective and constraints to the top level
        self.add_objective("cruise.aero_post.Torque", scaler=1.0)#0.1)
        self.add_constraint("geometry.duct_thickcon", lower=0.015,upper=0.15, scaler=1.0)#1.0)
        self.add_constraint("geometry.gap_con", equals=1.0)#1.0)
        #self.add_constraint("first_cst.cst_con", equals=0.0)
#        self.add_constraint("first_cst.cst_con", lower=0.1, upper=10.0, scaler=1.0)#1.0)
#        self.add_constraint("first_cst_2.cst_con2", lower=0.001, scaler=1.0)#1.0)


# =============================================================================
# OpenMDAO setup
# =============================================================================
prob = om.Problem()
prob.model = Top()

# Define Optimizer
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = "SNOPT"
prob.driver.opt_settings = {
    "Major feasibility tolerance": 1.0e-10,
    "Major optimality tolerance": 1.0e-10,
    "Minor feasibility tolerance": 1.0e-10,
    "Verify level": -1,
    "Function precision": 1.0e-8,
    "Major iterations limit": 500,
    "Nonderivative linesearch": None,
    "Print file": "opt_SNOPT_print.txt",
    "Summary file": "opt_SNOPT_summary.txt",
}
prob.driver.options["debug_print"] = ["nl_cons", "desvars", "objs"]
prob.driver.options["print_opt_prob"] = True
prob.driver.hist_file = "OptHist.hst"

# Define Recorder
prob.driver.recording_options["includes"] = []
prob.driver.recording_options["record_desvars"] = True
prob.driver.recording_options["record_objectives"] = True
prob.driver.recording_options["record_constraints"] = True
prob.driver.add_recorder(om.SqliteRecorder("OptHist.sql"))

# Setup Problem
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# ----------------------------- Run Optimization ---------------------------- #
if args.task == "run_driver":
    # Find Feasible Design
    # optFuncs = OptFuncs([daOptions], prob)
    # optFuncs.findFeasibleDesign(["cruise.aero_post.CL"], ["twist"], targets=[CL_target])

    prob.run_driver()

# -------------------------------- Run Primal ------------------------------- #
elif args.task == "run_model":
    # Find Feasible Design
    # optFuncs = OptFuncs([daOptions], prob)
    # optFuncs.findFeasibleDesign(["cruise.aero_post.CL"], ["twist"], targets=[CL_target])

    prob.run_model()

# ------------------------------- Check Totals ------------------------------ #
elif args.task == "check_totals":
    prob.run_model()
    prob.check_totals(method="fd", step=1e-3)

# --------------------------------- No Task --------------------------------- #
else:
    print("Task not found!")
    exit(0)
