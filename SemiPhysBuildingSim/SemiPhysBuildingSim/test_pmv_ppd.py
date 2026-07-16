from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments

# input variables
tdb = 27.5 # dry bulb air temperature, [$^{\circ}$C]
tr = tdb  # mean radiant temperature, [$^{\circ}$C]
v = 0.12  # average air speed, [m/s]
rh = 40  # relative humidity, [%]
activity = "Typing"  # participant's activity description
garments = ["Sweatpants", "T-shirt", "Shoes or sandals"]

met = met_typical_tasks[activity]  # activity met, [met]

print("met:"+str(met))
icl = sum(
    [clo_individual_garments[item] for item in garments]
)  # calculate total clothing insulation

# calculate the relative air velocity
vr = v_relative(v=v, met=met)
# calculate the dynamic clothing insulation
clo = clo_dynamic(clo=icl, met=met)

print("clo:"+str(clo))

# calculate PMV in accordance with the ASHRAE 55 2020
results = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, standard="ASHRAE")

# print the results
print(results)

# print PMV value
print(f"pmv={results['pmv']}, ppd={results['ppd']}%")