
from collections import defaultdict

# Data storage
leap_to_slide = {
    "LEAP005": "Slide_3",
    "LEAP008": "Slide_4",
    "LEAP009": "Slide_5",
    "LEAP010": "Slide_5",
    "LEAP011": "Slide_5",
    "LEAP012": "Slide_6",
    "LEAP013": "Slide_6",
    "LEAP014": "Slide_6",
    "LEAP015": "Slide_7",
    "LEAP017": "Slide_8",
    "LEAP021": "Slide_10",
    "LEAP026": "Slide_12",
    "LEAP028": "Slide_13",
    "LEAP030": "Slide_14",
    "LEAP032": "Slide_15",
    "LEAP034": "Slide_16",
    "LEAP036": "Slide_17",
    "LEAP037": "Slide_17",
    "LEAP038": "Slide_17",
    "LEAP039": "Slide_18",
    "LEAP041": "Slide_19",
    "LEAP042": "Slide_19",
    "LEAP043": "Slide_19",
    "LEAP044": "Slide_20",
    "LEAP046": "Slide_21",
    "LEAP048": "Slide_22",
    "LEAP050": "Slide_23",
    "LEAP064": "Slide_28",
    "LEAP066": "Slide_29",
    "LEAP067": "Slide_29",
    "LEAP068": "Slide_29",
    "LEAP069": "Slide_30",
    "LEAP071": "Slide_31",
    "LEAP073": "Slide_32",
    "LEAP075": "Slide_33",
    "LEAP076": "Slide_33",
    "LEAP078": "Slide_34",
    "LEAP080": "Slide_35",
    "LEAP082": "Slide_36",
    "LEAP083": "Slide_36",
    "LEAP084": "Slide_36",
    "LEAP085": "Slide_37",
    "LEAP086": "Slide_37",
    "LEAP087": "Slide_37",
    "LEAP088": "Slide_38",
    "LEAP090": "Slide_39",
    "LEAP092": "Slide_39",
    "LEAP093": "Slide_40",
    "LEAP095": "Slide_41",
    "LEAP103": "Slide_42",
    "LEAP104": "Slide_42",
    "LEAP105": "Slide_42",
    "LEAP106": "Slide_43",
    "LEAP110": "Slide_44",
    "LEAP114": "Slide_45",
    "LEAP115": "Slide_46",
    "LEAP118": "Slide_47",
    "LEAP120": "Slide_48",
    "LEAP128": "Slide_50",
    "LEAP130": "Slide_51",
    "LEAP135": "Slide_53",
    "LEAP138": "Slide_54"
}

# Reverse mapping for SLIDEID to LEAPID
slide_to_leap = defaultdict(list)
for leap_id, slide_id in leap_to_slide.items():
    slide_to_leap[slide_id].append(leap_id)

# Functions to retrieve data
def get_slide_id(leap_id):
    return leap_to_slide.get(leap_id, "Not Found")

def get_leap_ids(slide_id):
    return slide_to_leap.get(slide_id, [])

