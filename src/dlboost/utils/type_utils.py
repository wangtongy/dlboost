from jaxtyping import Complex, Float, Shaped
from torch import Tensor

KspaceData = Complex[Tensor, "length"] # 
KspaceSpokesData = Complex[Tensor, "spokes_num spoke_length"]
KspaceTraj = Float[Tensor, "2 length"]
KspaceSpokesTraj = Float[Tensor, "2 spokes_num spoke_length"]
Image2D = Float[Tensor, "h w"]
Image3D = Float[Tensor, "d h w"]
ComplexImage2D = Complex[Tensor, "h w"] | Image2D
ComplexImage3D = Shaped[ComplexImage2D, "d"]
MotionVectorField2D = Shaped[Image2D, "2"]
MotionVectorField3D = Shaped[Image3D, "3"]
