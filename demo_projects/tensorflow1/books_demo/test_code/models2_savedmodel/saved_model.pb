??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
?
ApplyGradientDescent
var"T?

alpha"T

delta"T
out"T?" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
.
Identity

input"T
output"T"	
Ttype
;
Maximum
x"T
y"T
z"T"
Ttype:

2	?
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
1
Square
x"T
y"T"
Ttype:

2	
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.11.02
b'unknown'??
P
model_inputPlaceholder*
dtype0*
shape:*
_output_shapes
:
O

real_valuePlaceholder*
dtype0*
shape:*
_output_shapes
:
]
random_normal/shapeConst*
valueB:*
dtype0*
_output_shapes
:
W
random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
random_normal/stddevConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
T0*
dtype0*
seed2 *

seed *
_output_shapes
:
w
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes
:
`
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes
:
r
weight
VariableV2*
dtype0*
shared_name *
shape:*
	container *
_output_shapes
:
?
weight/AssignAssignweightrandom_normal*
T0*
use_locking(*
validate_shape(*
_class
loc:@weight*
_output_shapes
:
_
weight/readIdentityweight*
T0*
_class
loc:@weight*
_output_shapes
:
R
zerosConst*
valueB*    *
dtype0*
_output_shapes
:
p
bias
VariableV2*
dtype0*
shared_name *
shape:*
	container *
_output_shapes
:
?
bias/AssignAssignbiaszeros*
T0*
use_locking(*
validate_shape(*
_class
	loc:@bias*
_output_shapes
:
Y
	bias/readIdentitybias*
T0*
_class
	loc:@bias*
_output_shapes
:
G
MulMulmodel_inputweight/read*
T0*
_output_shapes
:
=
addAddMul	bias/read*
T0*
_output_shapes
:
>
subSub
real_valueadd*
T0*
_output_shapes
:
8
SquareSquaresub*
T0*
_output_shapes
:
5
RankRankSquare*
T0*
_output_shapes
: 
M
range/startConst*
value	B : *
dtype0*
_output_shapes
: 
M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
_
rangeRangerange/startRankrange/delta*

Tidx0*#
_output_shapes
:?????????
Y
MeanMeanSquarerange*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ??*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
h
gradients/Mean_grad/ShapeShapeSquare*
T0*
out_type0*#
_output_shapes
:?????????
?
gradients/Mean_grad/SizeSizegradients/Mean_grad/Shape*
T0*
out_type0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
?
gradients/Mean_grad/addAddrangegradients/Mean_grad/Size*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/Shape_1Shapegradients/Mean_grad/mod*
T0*
out_type0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:
?
gradients/Mean_grad/range/startConst*
value	B : *
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
?
gradients/Mean_grad/range/deltaConst*
value	B :*
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
?
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*

Tidx0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/Fill/valueConst*
value	B :*
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
?
gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
T0*

index_type0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
T0*
N*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
?
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
?
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*
T0*

Tmultiples0*
_output_shapes
:
j
gradients/Mean_grad/Shape_2ShapeSquare*
T0*
out_type0*#
_output_shapes
:?????????
^
gradients/Mean_grad/Shape_3Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
a
gradients/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
?
gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
T0*
_output_shapes
: 
?
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 
}
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*
_output_shapes
:
~
gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
e
gradients/Square_grad/MulMulsubgradients/Square_grad/Const*
T0*
_output_shapes
:
}
gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*
T0*
_output_shapes
:
k
gradients/sub_grad/ShapeShape
real_value*
T0*
out_type0*#
_output_shapes
:?????????
f
gradients/sub_grad/Shape_1Shapeadd*
T0*
out_type0*#
_output_shapes
:?????????
?
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
gradients/sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:
?
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
?
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
:
?
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
_output_shapes
:
d
gradients/add_grad/ShapeShapeMul*
T0*
out_type0*#
_output_shapes
:?????????
d
gradients/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
?
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/add_grad/SumSum-gradients/sub_grad/tuple/control_dependency_1(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
gradients/add_grad/Sum_1Sum-gradients/sub_grad/tuple/control_dependency_1*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
?
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*
_output_shapes
:
?
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
l
gradients/Mul_grad/ShapeShapemodel_input*
T0*
out_type0*#
_output_shapes
:?????????
d
gradients/Mul_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
?
(gradients/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_grad/Shapegradients/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
z
gradients/Mul_grad/MulMul+gradients/add_grad/tuple/control_dependencyweight/read*
T0*
_output_shapes
:
?
gradients/Mul_grad/SumSumgradients/Mul_grad/Mul(gradients/Mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients/Mul_grad/ReshapeReshapegradients/Mul_grad/Sumgradients/Mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
|
gradients/Mul_grad/Mul_1Mulmodel_input+gradients/add_grad/tuple/control_dependency*
T0*
_output_shapes
:
?
gradients/Mul_grad/Sum_1Sumgradients/Mul_grad/Mul_1*gradients/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients/Mul_grad/Reshape_1Reshapegradients/Mul_grad/Sum_1gradients/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/Mul_grad/tuple/group_depsNoOp^gradients/Mul_grad/Reshape^gradients/Mul_grad/Reshape_1
?
+gradients/Mul_grad/tuple/control_dependencyIdentitygradients/Mul_grad/Reshape$^gradients/Mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Mul_grad/Reshape*
_output_shapes
:
?
-gradients/Mul_grad/tuple/control_dependency_1Identitygradients/Mul_grad/Reshape_1$^gradients/Mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Mul_grad/Reshape_1*
_output_shapes
:
b
GradientDescent/learning_rateConst*
valueB
 *
?#<*
dtype0*
_output_shapes
: 
?
2GradientDescent/update_weight/ApplyGradientDescentApplyGradientDescentweightGradientDescent/learning_rate-gradients/Mul_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class
loc:@weight*
_output_shapes
:
?
0GradientDescent/update_bias/ApplyGradientDescentApplyGradientDescentbiasGradientDescent/learning_rate-gradients/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class
	loc:@bias*
_output_shapes
:

GradientDescentNoOp1^GradientDescent/update_bias/ApplyGradientDescent3^GradientDescent/update_weight/ApplyGradientDescent
*
initNoOp^bias/Assign^weight/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
m
save/SaveV2/tensor_namesConst*!
valueBBbiasBweight*
dtype0*
_output_shapes
:
g
save/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0*
_output_shapes
:
y
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbiasweight*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 

save/RestoreV2/tensor_namesConst"/device:CPU:0*!
valueBBbiasBweight*
dtype0*
_output_shapes
:
y
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0*
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes

::
?
save/AssignAssignbiassave/RestoreV2*
T0*
use_locking(*
validate_shape(*
_class
	loc:@bias*
_output_shapes
:
?
save/Assign_1Assignweightsave/RestoreV2:1*
T0*
use_locking(*
validate_shape(*
_class
loc:@weight*
_output_shapes
:
6
save/restore_allNoOp^save/Assign^save/Assign_1
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
?
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_3a075f976c3c4930b1616945db3f84e1/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
~
save_1/SaveV2/tensor_namesConst"/device:CPU:0*!
valueBBbiasBweight*
dtype0*
_output_shapes
:
x
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0*
_output_shapes
:
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbiasweight"/device:CPU:0*
dtypes
2
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*

axis *
T0*
N*
_output_shapes
:
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*!
valueBBbiasBweight*
dtype0*
_output_shapes
:
{
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0*
_output_shapes
:
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes

::
?
save_1/AssignAssignbiassave_1/RestoreV2*
T0*
use_locking(*
validate_shape(*
_class
	loc:@bias*
_output_shapes
:
?
save_1/Assign_1Assignweightsave_1/RestoreV2:1*
T0*
use_locking(*
validate_shape(*
_class
loc:@weight*
_output_shapes
:
>
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"{
	variablesnl
;
weight:0weight/Assignweight/read:02random_normal:08
-
bias:0bias/Assignbias/read:02zeros:08"?
trainable_variablesnl
;
weight:0weight/Assignweight/read:02random_normal:08
-
bias:0bias/Assignbias/read:02zeros:08"
train_op

GradientDescent*v
graph_informationa
$
model_input
model_input:0
model_output
add:0tensorflow/serving/predict