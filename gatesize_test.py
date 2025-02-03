#BSD 3-Clause License
#
#Copyright (c) 2024, ASU-VDA-Lab
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import openroad as ord
from openroad import Tech, Design, Timing
import os, odb
from pathlib import Path
from OpenROAD_helper import *
import argparse

parser = argparse.ArgumentParser(description = "parsing the name of the benchmark")
parser.add_argument("--design_name", type = str, default = "NV_NVDLA_partition_m")
pyargs = parser.parse_args()
##############################################
# Load the design using OpenROAD Python APIs #
##############################################
print("*********************Load design, lib, lef, and sdc********************")
print("*****Please go through the code to see the detailed implementation*****")
tech, design = load_design(pyargs.design_name, False)
timing = Timing(design)
db = ord.get_db()
corner = timing.getCorners()[0]
block = design.getBlock()

#########################################
# Get pins, insts, and nets from OpenDB #
#########################################
print("*******************Get pins, insts, and nets from OpenDB***************")
print("*****Please go through the code to see the detailed implementation*****")
pins = block.getITerms()
insts = block.getInsts()
nets = block.getNets()

# Other ways to get pins from OpenDB
# One way is iterating through insts
for inst in insts:
  inst_ITerms = inst.getITerms()
  for pin in inst_ITerms:
    pass
# The other way is iterating through nets
for net in nets:
  net_ITerms = net.getITerms()
  for pin in net_ITerms:
    pass

#########################
# Get all library cells #
#########################
print("************************Get all library cells**************************")
print("*****Please go through the code to see the detailed implementation*****")
libs = db.getLibs()
for lib in libs:
  for master in lib.getMasters():
    libcell_name = master.getName()
    libcell_area = master.getHeight() * master.getWidth()

'''
#######################################################################
# How to use the name of the instance to get the instance from OpenDB #
#######################################################################
inst = block.findInst("u_NV_NVDLA_cmac_u_core_u_mac_1_mul_124_55_g84957")
print("-------------The instance we get-------------")
print(inst.getName())

#########################################################
# Check if the instance is a macro or a sequential cell #
#########################################################
if design.isSequential(inst.getMaster()):
  print("It's a sequential cell!")
if inst.getMaster().isBlock():
  print("It's a macro!")

###############################################################################
# How to use the name of the library cell to get the library cell from OpenDB #
###############################################################################
master = db.findMaster("AOI22xp5_ASAP7_75t_R")
print("-----------The library cell we get-----------")
print(master.getName())

#########################################################################
# How to get timing information (pin slew, pin slack, pin arrival time) #
#########################################################################
print("*****get pin's timing information*****")
# Use the name of the instance to find the instance
inst = block.findInst("u_NV_NVDLA_cmac_u_core_u_mac_1_mul_124_55_g84957")
pins = inst.getITerms()
for pin in pins:
  # Filter out pins connecting to constant 1 or 0
  if pin.getNet() != None:
    # Filter out the VDD/VSS pin
    if pin.getNet().getSigType() != 'POWER' and pin.getNet().getSigType() != 'GROUND':
      pin_tran = timing.getPinSlew(pin)
      pin_slack = min(timing.getPinSlack(pin, timing.Fall, timing.Max), timing.getPinSlack(pin, timing.Rise, timing.Max))
      pin_rise_arr = timing.getPinArrival(pin, timing.Rise)
      pin_fall_arr = timing.getPinArrival(pin, timing.Fall)
      if pin.isInputSignal():
        input_pin_cap = timing.getPortCap(pin, corner, timing.Max)
      else:
        input_pin_cap = -1
      # This gives the sum of the loading pins' capacitance
      output_load_pin_cap = get_output_load_pin_cap(pin, corner, timing)
      # This will add net's capacitance to the output load capacitance
      output_load_cap = timing.getNetCap(pin.getNet(), corner, timing.Max) if pin.isOutputSignal() else -1


      # We have to use MTerm (library cell's pin, ITerms are instances' pins) to get the constraints
      # such as max load cap and max slew (pin transition time)
      library_cell_pin = [MTerm for MTerm in pin.getInst().getMaster().getMTerms() if (pin.getInst().getName() + "/" + MTerm.getName()) == pin.getName()][0]
      pin_tran_limit = timing.getMaxSlewLimit(library_cell_pin)
      output_load_cap_limit = timing.getMaxCapLimit(library_cell_pin)
      print("""Pin name: %s
Pin transition time: %.25f
Pin's maximum available transition time: %.25f
Pin slack: %.25f
Pin rising arrival time: %.25f
Pin falling arrival time: %.25f
Pin's input capacitance: %.25f
Pin's output pin capacitance: %.25f
Pin's output capacitance: %.25f
Pin's maximum available output capacitance: %.25f
-------------------------------"""%(
      design.getITermName(pin),
      pin_tran,
      pin_tran_limit,
      pin_slack,
      pin_rise_arr,
      pin_fall_arr,
      input_pin_cap,
      output_load_pin_cap,
      output_load_cap,
      output_load_cap_limit))

#####################################################
# How to get power information (static and dynamic) #
#####################################################
print("*****get instance's power information*****")
leakage = timing.staticPower(inst, corner)
internal_and_switching = timing.dynamicPower(inst, corner)
print("""Instance name: %s
Leakage power: %.25f
Internal power + switching power: %.25f
-------------------------------"""%(
  inst.getName(),
  leakage,
  internal_and_switching))

##############################
# How to perform gate sizing #
##############################
timing.makeEquivCells()
# First pick an instance
inst = block.findInst("u_NV_NVDLA_cmac_u_core_u_mac_1_mul_124_55_g84957")
# Then get the library cell information
inst_master = inst.getMaster()
print("-----------Reference library cell-----------")
print(inst_master.getName())
print("-----Library cells with different sizes-----")
equiv_cells = timing.equivCells(inst_master)
for equiv_cell in equiv_cells:
  print(equiv_cell.getName())
'''

"""
# Reset the timing graph before performing gate sizing
timing.resetTiming()

# Perform gate sizing
inst.swapMaster(equiv_cells[2])
print("----Change to the following library cell----")
print(inst.getMaster().getName())
"""
#####################################
# Perform Legalization after sizing #
#####################################
site = design.getBlock().getRows()[0].getSite()
max_disp_x = int(design.micronToDBU(0.1) / site.getWidth())
max_disp_y = int(design.micronToDBU(0.1) / site.getHeight())
design.getOpendp().detailedPlacement(max_disp_x, max_disp_y, "", False)

##############################################
# Run Global Route After Legalization and    #
# Get The Updated RC Info. for Timing Update #
##############################################
signal_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
signal_high_layer = design.getTech().getDB().getTech().findLayer("M7").getRoutingLevel()
clk_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
clk_high_layer = design.getTech().getDB().getTech().findLayer("M7").getRoutingLevel()
grt = design.getGlobalRouter()
grt.clear()
grt.setAllowCongestion(True)
grt.setMinRoutingLayer(signal_low_layer)
grt.setMaxRoutingLayer(signal_high_layer)
grt.setMinLayerForClock(clk_low_layer)
grt.setMaxLayerForClock(clk_high_layer)
grt.setAdjustment(0.5)
grt.setVerbose(False)
grt.globalRoute(False)
design.evalTclString("estimate_parasitics -global_routing")

'''
########################################################
# Timing information will be updated in the background #
########################################################
print("*****get pin's timing information after gate sizing*****")
for pin in pins:
  # Filter out pins connecting to constant 1 or 0
  if pin.getNet() != None:
    # Filter out the VDD/VSS pin
    if pin.getNet().getSigType() != 'POWER' and pin.getNet().getSigType() != 'GROUND':
      pin_tran = timing.getPinSlew(pin)
      pin_slack = min(timing.getPinSlack(pin, timing.Fall, timing.Max), timing.getPinSlack(pin, timing.Rise, timing.Max))
      pin_rise_arr = timing.getPinArrival(pin, timing.Rise)
      pin_fall_arr = timing.getPinArrival(pin, timing.Fall)
      if pin.isInputSignal():
        input_pin_cap = timing.getPortCap(pin, corner, timing.Max)
      else:
        input_pin_cap = -1
      # This gives the sum of the loading pins' capacitance
      output_load_pin_cap = get_output_load_pin_cap(pin, corner, timing)
      # This will add net's capacitance to the output load capacitance
      output_load_cap = timing.getNetCap(pin.getNet(), corner, timing.Max) if pin.isOutputSignal() else -1
      print("""Pin name: %s
Pin transition time: %.25f
Pin slack: %.25f
Pin rising arrival time: %.25f
Pin falling arrival time: %.25f
Pin's input capacitance: %.25f
Pin's output pin capacitance: %.25f
Pin's output capacitance: %.25f
-------------------------------"""%(
      design.getITermName(pin),
      pin_tran,
      pin_slack,
      pin_rise_arr,
      pin_fall_arr,
      input_pin_cap,
      output_load_pin_cap,
      output_load_cap))
'''

# Save the design original .size file
with open(f'{pyargs.design_name}_original.size', 'w') as file:
    for inst in design.getBlock().getInsts():
        file.write(f"{inst.getName()} {inst.getMaster().getName()}\n")

set_dont_touch = True

# Set the dont_touch attribute to the instances that are sequential or are macros
if set_dont_touch:
  dont_touch_cells = []
  for inst in insts:
    if design.isSequential(inst.getMaster()) or inst.getMaster().isBlock():
      dont_touch_cells.append(inst.getName())
  print("Number of dont touch cells: ", len(dont_touch_cells))
  print("Dont touch cells (Examples): ", dont_touch_cells[:5])
  #design.evalTclString("set_dont_touch %s"%(" ".join(dont_touch_cells)))
  #design.evalTclString("report_dont_touch")
  for dont in dont_touch_cells:
     design.evalTclString("set_dont_touch %s"%dont)
  #for dont in dont_touch_cells:
    #db.dontTouch(dont)
  #db.dont


# Run gate sizing
# Before gate sizing - store initial leakage power
leakageBeforeSwap = 0
for inst in design.getBlock().getInsts():
    leakageBeforeSwap += timing.staticPower(inst, corner)
leakageBeforeSwap *= 1000000 # Convert to uW

# Get the pre gatesizing timing information 
design.evalTclString("report_checks")
design.evalTclString("report_tns") 
design.evalTclString("report_wns")

import time
start = time.time()

# Run gate sizing
design.evalTclString("repair_design -max_wire_length 100 -slew_margin 0 -cap_margin 0")
# design.evalTclString("repair_timing -recover_power 100 -verbose")
# design.evalTclString("repair_timing -setup -hold -recover_power 100 -skip_pin_swap -skip_gate_cloning -skip_buffering -skip_buffer_removal -max_buffer_percent 0 -verbose")
repair_design_time = time.time()
print(f"repair_design took {repair_design_time - start} seconds")

#####################################
# Perform Legalization after sizing #
#####################################
site = design.getBlock().getRows()[0].getSite()
max_disp_x = int(design.micronToDBU(0.1) / site.getWidth())
max_disp_y = int(design.micronToDBU(0.1) / site.getHeight())
design.getOpendp().detailedPlacement(max_disp_x, max_disp_y, "", False)

post_repair_design_time = time.time()
print(f"Detailed Placement after repair_design took {time.time() - repair_design_time} seconds")

##############################################
# Run Global Route After Legalization and    #
# Get The Updated RC Info. for Timing Update #
##############################################
signal_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
signal_high_layer = design.getTech().getDB().getTech().findLayer("M7").getRoutingLevel()
clk_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
clk_high_layer = design.getTech().getDB().getTech().findLayer("M7").getRoutingLevel()
grt = design.getGlobalRouter()
grt.clear()
grt.setAllowCongestion(True)
grt.setMinRoutingLayer(signal_low_layer)
grt.setMaxRoutingLayer(signal_high_layer)
grt.setMinLayerForClock(clk_low_layer)
grt.setMaxLayerForClock(clk_high_layer)
grt.setAdjustment(0.5)
grt.setVerbose(False)
grt.globalRoute(False)
design.evalTclString("estimate_parasitics -global_routing")

post_global_route_time = time.time()
print(f"Global Route after detailed placement took {post_global_route_time - post_repair_design_time} seconds")

# Get the updated timing information 
design.evalTclString("report_checks")
design.evalTclString("report_tns") 
design.evalTclString("report_wns")

print("After repair_sizing that fixes slew and cap violations")

# Start Evaluation
tns, slew, cap, leakage = 0, 0, 0, 0
# Penalties are subject to change
tnsPenalty, slewPenalty, capPenalty = 10, 20, 20

# Get all timing metrics
design.evalTclString("report_tns > tns_evaluation_temp.txt")
with open("tns_evaluation_temp.txt", "r") as file:
    for line in file:
        tns = float(line.split()[1]) / 1000

for pin_ in design.getBlock().getITerms():
    if pin_.getNet() != None:
        if pin_.getNet().getSigType() not in ['POWER', 'GROUND', 'CLOCK']:
            library_cell_pin = [MTerm for MTerm in pin_.getInst().getMaster().getMTerms() 
                              if (pin_.getInst().getName() + "/" + MTerm.getName()) == pin_.getName()][0]       
            
            # Check slew violations
            if timing.getMaxSlewLimit(library_cell_pin) < timing.getPinSlew(pin_):
                diff = abs(timing.getMaxSlewLimit(library_cell_pin) - timing.getPinSlew(pin_)) * 1e9
                slew += diff
                
            # Check capacitance violations for output pins
            if pin_.isOutputSignal():
                if timing.getMaxCapLimit(library_cell_pin) < timing.getNetCap(pin_.getNet(), corner, timing.Max):
                    diff = abs(timing.getMaxCapLimit(library_cell_pin) - 
                             timing.getNetCap(pin_.getNet(), corner, timing.Max)) * 1e15
                    cap += diff

os.remove("tns_evaluation_temp.txt")

# Calculate leakage power difference
leakage = 0
for inst in design.getBlock().getInsts():
    leakage += timing.staticPower(inst, corner)
leakage *= 1000000  # Convert to uW
leakage -= leakageBeforeSwap

# Adjust penalties
tnsPenalty = 0 if tns >= 0.0 else tnsPenalty
slewPenalty = 0 if slew == 0 else slewPenalty  
capPenalty = 0 if cap == 0 else capPenalty

# Calculate final score
score = leakage + tnsPenalty * abs(tns) + slewPenalty * abs(slew) + capPenalty * abs(cap)

# Print results
print("===================================================")
print(f"TNS: {tns} ns")
print("No slew violation" if slewPenalty == 0 else f"Total slew violation difference: {slew} ns")
print("No load capacitance violation" if capPenalty == 0 else f"Total load capacitance violation difference: {cap} fF")
print(f"Leakage power difference: {leakage} uW")
print(f"Score: {score}")
print("Require runtime in official score calculation")
print("===================================================")


pre_repair_timing_time = time.time()
# Also run repair_timing again
design.evalTclString("repair_timing -skip_buffering -skip_pin_swap -skip_gate_cloning -skip_buffer_removal")
post_repair_timing_time = time.time()
print(f"repair_timing took {post_repair_timing_time - pre_repair_timing_time} seconds")
#####################################
# Perform Legalization after sizing #
#####################################
site = design.getBlock().getRows()[0].getSite()
max_disp_x = int(design.micronToDBU(0.1) / site.getWidth())
max_disp_y = int(design.micronToDBU(0.1) / site.getHeight())
design.getOpendp().detailedPlacement(max_disp_x, max_disp_y, "", False)

post_repair_timing_placement_time = time.time()
print(f"Detailed Placement after repair_timing took {post_repair_timing_placement_time - post_repair_timing_time} seconds")

##############################################
# Run Global Route After Legalization and    #
# Get The Updated RC Info. for Timing Update #
##############################################
signal_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
signal_high_layer = design.getTech().getDB().getTech().findLayer("M7").getRoutingLevel()
clk_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
clk_high_layer = design.getTech().getDB().getTech().findLayer("M7").getRoutingLevel()
grt = design.getGlobalRouter()
grt.clear()
grt.setAllowCongestion(True)
grt.setMinRoutingLayer(signal_low_layer)
grt.setMaxRoutingLayer(signal_high_layer)
grt.setMinLayerForClock(clk_low_layer)
grt.setMaxLayerForClock(clk_high_layer)
grt.setAdjustment(0.5)
grt.setVerbose(False)
grt.globalRoute(False)
design.evalTclString("estimate_parasitics -global_routing")

post_repair_timing_global_route_time = time.time()
print(f"Global Route after detailed placement took {post_repair_timing_global_route_time - post_repair_timing_placement_time} seconds")

print(f"Total sizing iteration took {post_repair_timing_global_route_time - pre_repair_timing_time + post_global_route_time - start} seconds")

# Get the updated timing information 
design.evalTclString("report_checks")
design.evalTclString("report_tns") 
design.evalTclString("report_wns")

print("After repair_timing that fixes tns")


# Start Evaluation
tns, slew, cap, leakage = 0, 0, 0, 0
# Penalties are subject to change
tnsPenalty, slewPenalty, capPenalty = 10, 20, 20

# Get all timing metrics
design.evalTclString("report_tns > tns_evaluation_temp.txt")
with open("tns_evaluation_temp.txt", "r") as file:
    for line in file:
        tns = float(line.split()[1]) / 1000

for pin_ in design.getBlock().getITerms():
    if pin_.getNet() != None:
        if pin_.getNet().getSigType() not in ['POWER', 'GROUND', 'CLOCK']:
            library_cell_pin = [MTerm for MTerm in pin_.getInst().getMaster().getMTerms() 
                              if (pin_.getInst().getName() + "/" + MTerm.getName()) == pin_.getName()][0]       
            
            # Check slew violations
            if timing.getMaxSlewLimit(library_cell_pin) < timing.getPinSlew(pin_):
                diff = abs(timing.getMaxSlewLimit(library_cell_pin) - timing.getPinSlew(pin_)) * 1e9
                slew += diff
                
            # Check capacitance violations for output pins
            if pin_.isOutputSignal():
                if timing.getMaxCapLimit(library_cell_pin) < timing.getNetCap(pin_.getNet(), corner, timing.Max):
                    diff = abs(timing.getMaxCapLimit(library_cell_pin) - 
                             timing.getNetCap(pin_.getNet(), corner, timing.Max)) * 1e15
                    cap += diff

os.remove("tns_evaluation_temp.txt")

# Calculate leakage power difference
leakage = 0
for inst in design.getBlock().getInsts():
    leakage += timing.staticPower(inst, corner)
leakage *= 1000000  # Convert to uW
leakage -= leakageBeforeSwap

# Adjust penalties
tnsPenalty = 0 if tns >= 0.0 else tnsPenalty
slewPenalty = 0 if slew == 0 else slewPenalty  
capPenalty = 0 if cap == 0 else capPenalty

# Calculate final score
score = leakage + tnsPenalty * abs(tns) + slewPenalty * abs(slew) + capPenalty * abs(cap)

# Print results
print("===================================================")
print(f"TNS: {tns} ns")
print("No slew violation" if slewPenalty == 0 else f"Total slew violation difference: {slew} ns")
print("No load capacitance violation" if capPenalty == 0 else f"Total load capacitance violation difference: {cap} fF")
print(f"Leakage power difference: {leakage} uW")
print(f"Score: {score}")
print("Require runtime in official score calculation")
print("===================================================")


# Save the updated .size file
# Size file format: <cell_name> <lib_cell_name>
with open(f"{pyargs.design_name}_updated.size", "w") as file:
    for inst in design.getBlock().getInsts():
        file.write(f"{inst.getName()} {inst.getMaster().getName()}\n")


# design.evalTclString("get_endpoints_and_critical_paths")

###########################################
# How to dump the updated design DEF file #
###########################################
odb.write_def(block, "temp.def")
