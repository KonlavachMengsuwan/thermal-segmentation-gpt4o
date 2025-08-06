# 1, import from installed library
from pydantic import (BaseModel,
                      PlainSerializer,
                      PlainValidator
                      )

# 2, import from source code
import src._gen_mask as _gen_mask
import src._modify_mask as _modify_mask
import src._gen_temp as _gen_temp
import src._gen_graph as _gen_graph

# 3, import from python library
from typing import (Any,
                    #Optional,
                    Annotated
                    )

#################################################################
"""
for pydantic
https://zenn.dev/yosemat/articles/6834cfc8de0d86
"""


def validate(_: Any):
    ...


def serialize(_: Any):
    ...

Gen_Mask_Annotated = Annotated[_gen_mask.Gen_Mask, PlainValidator(validate), PlainSerializer(serialize)]
Modify_Mask_Annotated = Annotated[_modify_mask.Modify_Mask, PlainValidator(validate), PlainSerializer(serialize)]
Gen_Temp_Annotated = Annotated[_gen_temp.Gen_Temp, PlainValidator(validate), PlainSerializer(serialize)]
Gen_Graph_Annotated = Annotated[_gen_graph.Gen_Graph, PlainValidator(validate), PlainSerializer(serialize)]

#################################################################


class GlobalVariable(BaseModel):

    path : str = ""
    mask_type : str = ""
    use_pickle : bool = False
    min_area_percent : float = 2
    use_temp : bool = False
    filename_ext : str = ""

    common_filename : str = ""

    gen_mask : Gen_Mask_Annotated = None
    modify_mask : Modify_Mask_Annotated = None
    gen_temp : Gen_Temp_Annotated = None
    gen_graph : Gen_Graph_Annotated = None

    mask_color_array2 : list = []

    def __init__(self, opt):
        super().__init__(path = opt.path,
                         mask_type = opt.mask_type,
                         use_pickle = opt.use_pickle,
                         min_area_percent = opt.min_area_percent,
                         use_temp = opt.use_temp,
                         filename_ext = opt.filename_ext
                         )
        self.gen_mask = _gen_mask.Gen_Mask_SAM()
        self.modify_mask = _modify_mask.Modify_Mask_Default()
        self.gen_temp = _gen_temp.Gen_Temp_CSV()
        self.gen_graph = _gen_graph.Gen_Graph_BoxPlot()

        for color in const.GEN_TEMP_MASK_COLOR_ARRAY:
            color_value = '#' + format(color[0], '02x') + format(color[1], '02x') + format(color[2], '02x') + '80'
            self.mask_color_array2.append(color_value)

    @classmethod
    def generate_instance(cls, opt):
        return cls(opt)


class ConstVariable(BaseModel, frozen = True):
    MIN_AREA_RATIO_TH : float = 0.25

    #GEN_MASK_SAM_DEVICE : str = "cuda"
    GEN_MASK_SAM_DEVICE : str = "cpu"
    #GEN_MASK_SAM_CHECKPOINT : str = "../sam_vt/sam_vit_h_4b8939.pth"
    GEN_MASK_SAM_CHECKPOINT : str = "./sam_vit_h_4b8939.pth"
    GEN_MASK_SAM_MODEL_TYPE: str = "vit_h"

    # R, G, B
    GEN_TEMP_MASK_COLOR_ARRAY: list = \
        [[0, 0, 255],
         [0, 255, 0],
         [255, 0, 0],
         [0, 255, 255],
         [255, 0, 255],
         [255, 255, 0],

         [0, 0, 128],
         [0, 128, 0],
         [128, 0, 0],
         [0, 128, 128],
         [128, 0, 128],
         [128, 128, 0],

         [135, 206, 250],
         [140, 238, 144],
         [240, 128, 128],
         [224, 255, 255],
         [255, 182, 193],
         [255, 254, 224],

         [0, 0, 139],
         [0, 100, 0],
         [139, 0, 0],
         [0, 139, 139],
         [139, 0, 139],
         [255, 140, 0]
         ]

    def __init__(self):
        super().__init__()

    @classmethod
    def generate_instance(cls):
        return cls()

const : ConstVariable = None
var : GlobalVariable = None
