│█
┐г
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02unknown8■╘
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:  *
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
: *
dtype0
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

: *
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:*
dtype0
{
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_19/kernel
t
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes
:	А*
dtype0
s
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_19/bias
l
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes	
:А*
dtype0
П
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_6/gamma
И
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_6/beta
Ж
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_6/moving_mean
Ф
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:А*
dtype0
г
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_6/moving_variance
Ь
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:А*
dtype0
{
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@* 
shared_namedense_20/kernel
t
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes
:	А@*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:@*
dtype0
О
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_7/gamma
З
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_7/beta
Е
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_7/moving_mean
У
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:@*
dtype0
в
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_7/moving_variance
Ы
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:@*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	@А*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:А*
dtype0
П
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_8/gamma
И
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_8/beta
Ж
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_8/moving_mean
Ф
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:А*
dtype0
г
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_8/moving_variance
Ь
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:А*
dtype0
{
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_22/kernel
t
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes
:	А*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:*
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:*
dtype0

NoOpNoOp
еA
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*р@
value╓@B╙@ B╠@
Я
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
Ч
axis
	gamma
 beta
!moving_mean
"moving_variance
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
Ч
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2regularization_losses
3trainable_variables
4	variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
Ч
<axis
	=gamma
>beta
?moving_mean
@moving_variance
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
h

Ekernel
Fbias
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
Ч
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
h

Tkernel
Ubias
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
h

Zkernel
[bias
\regularization_losses
]trainable_variables
^	variables
_	keras_api
 
ж
0
1
2
3
4
 5
'6
(7
.8
/9
610
711
=12
>13
E14
F15
L16
M17
T18
U19
Z20
[21
ц
0
1
2
3
4
 5
!6
"7
'8
(9
.10
/11
012
113
614
715
=16
>17
?18
@19
E20
F21
L22
M23
N24
O25
T26
U27
Z28
[29
н
regularization_losses
trainable_variables
`non_trainable_variables
	variables
ametrics
blayer_metrics

clayers
dlayer_regularization_losses
 
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_17/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
regularization_losses
trainable_variables
enon_trainable_variables
	variables
fmetrics
glayer_metrics

hlayers
ilayer_regularization_losses
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
regularization_losses
trainable_variables
jnon_trainable_variables
	variables
kmetrics
llayer_metrics

mlayers
nlayer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
!2
"3
н
#regularization_losses
$trainable_variables
onon_trainable_variables
%	variables
pmetrics
qlayer_metrics

rlayers
slayer_regularization_losses
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
н
)regularization_losses
*trainable_variables
tnon_trainable_variables
+	variables
umetrics
vlayer_metrics

wlayers
xlayer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
02
13
н
2regularization_losses
3trainable_variables
ynon_trainable_variables
4	variables
zmetrics
{layer_metrics

|layers
}layer_regularization_losses
[Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
░
8regularization_losses
9trainable_variables
~non_trainable_variables
:	variables
metrics
Аlayer_metrics
Бlayers
 Вlayer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
?2
@3
▓
Aregularization_losses
Btrainable_variables
Гnon_trainable_variables
C	variables
Дmetrics
Еlayer_metrics
Жlayers
 Зlayer_regularization_losses
[Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_21/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

E0
F1
▓
Gregularization_losses
Htrainable_variables
Иnon_trainable_variables
I	variables
Йmetrics
Кlayer_metrics
Лlayers
 Мlayer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
N2
O3
▓
Pregularization_losses
Qtrainable_variables
Нnon_trainable_variables
R	variables
Оmetrics
Пlayer_metrics
Рlayers
 Сlayer_regularization_losses
[Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_22/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

T0
U1
▓
Vregularization_losses
Wtrainable_variables
Тnon_trainable_variables
X	variables
Уmetrics
Фlayer_metrics
Хlayers
 Цlayer_regularization_losses
\Z
VARIABLE_VALUEdense_23/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_23/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Z0
[1

Z0
[1
▓
\regularization_losses
]trainable_variables
Чnon_trainable_variables
^	variables
Шmetrics
Щlayer_metrics
Ъlayers
 Ыlayer_regularization_losses
8
!0
"1
02
13
?4
@5
N6
O7
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
 
 
 
 
 
 
 
 
 
 
 

!0
"1
 
 
 
 
 
 
 
 
 

00
11
 
 
 
 
 
 
 
 
 

?0
@1
 
 
 
 
 
 
 
 
 

N0
O1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_4Placeholder*'
_output_shapes
:          *
dtype0*
shape:          
┘
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4dense_17/kerneldense_17/biasdense_18/kerneldense_18/bias%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/betadense_19/kerneldense_19/bias%batch_normalization_6/moving_variancebatch_normalization_6/gamma!batch_normalization_6/moving_meanbatch_normalization_6/betadense_20/kerneldense_20/bias%batch_normalization_7/moving_variancebatch_normalization_7/gamma!batch_normalization_7/moving_meanbatch_normalization_7/betadense_21/kerneldense_21/bias%batch_normalization_8/moving_variancebatch_normalization_8/gamma!batch_normalization_8/moving_meanbatch_normalization_8/betadense_22/kerneldense_22/biasdense_23/kerneldense_23/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_10339
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
№
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpConst*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_11375
Я
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_17/kerneldense_17/biasdense_18/kerneldense_18/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense_19/kerneldense_19/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancedense_20/kerneldense_20/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancedense_21/kerneldense_21/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancedense_22/kerneldense_22/biasdense_23/kerneldense_23/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_11475∙┤
г
к
B__inference_dense_17_layer_call_and_return_conditional_losses_9588

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╕
и
5__inference_batch_normalization_8_layer_call_fn_11212

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_95622
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
█
}
(__inference_dense_22_layer_call_fn_11237

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_22_layer_call_and_return_conditional_losses_98672
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
д
л
C__inference_dense_17_layer_call_and_return_conditional_losses_10796

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
а
л
C__inference_dense_19_layer_call_and_return_conditional_losses_10918

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddV
ExpExpBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Exp\
IdentityIdentityExp:y:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
тв
╙
G__inference_functional_7_layer_call_and_return_conditional_losses_10529

inputs+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource/
+batch_normalization_5_assignmovingavg_103641
-batch_normalization_5_assignmovingavg_1_10370?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource/
+batch_normalization_6_assignmovingavg_104031
-batch_normalization_6_assignmovingavg_1_10409?
;batch_normalization_6_batchnorm_mul_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource/
+batch_normalization_7_assignmovingavg_104421
-batch_normalization_7_assignmovingavg_1_10448?
;batch_normalization_7_batchnorm_mul_readvariableop_resource;
7batch_normalization_7_batchnorm_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource/
+batch_normalization_8_assignmovingavg_104801
-batch_normalization_8_assignmovingavg_1_10486?
;batch_normalization_8_batchnorm_mul_readvariableop_resource;
7batch_normalization_8_batchnorm_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource
identityИв9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpи
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_17/MatMul/ReadVariableOpО
dense_17/MatMulMatMulinputs&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_17/MatMulз
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_17/BiasAdd/ReadVariableOpе
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_17/BiasAddp
dense_17/EluEludense_17/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_17/Eluи
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_18/MatMul/ReadVariableOpв
dense_18/MatMulMatMuldense_17/Elu:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_18/MatMulз
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_18/BiasAdd/ReadVariableOpе
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_18/BiasAddp
dense_18/ExpExpdense_18/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_18/Exp╢
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_5/moments/mean/reduction_indices█
"batch_normalization_5/moments/meanMeandense_18/Exp:y:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_5/moments/mean╛
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_5/moments/StopGradientЁ
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_18/Exp:y:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:         21
/batch_normalization_5/moments/SquaredDifference╛
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_5/moments/variance/reduction_indicesК
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_5/moments/variance┬
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_5/moments/Squeeze╩
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1▀
+batch_normalization_5/AssignMovingAvg/decayConst*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/10364*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_5/AssignMovingAvg/decay╘
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_5_assignmovingavg_10364*
_output_shapes
:*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOp░
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/10364*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/subз
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/10364*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/mulГ
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_5_assignmovingavg_10364-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/10364*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpх
-batch_normalization_5/AssignMovingAvg_1/decayConst*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/10370*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_5/AssignMovingAvg_1/decay┌
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_5_assignmovingavg_1_10370*
_output_shapes
:*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp║
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/10370*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/sub▒
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/10370*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/mulП
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_5_assignmovingavg_1_10370/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/10370*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_5/batchnorm/add/y┌
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/addе
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtр
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOp▌
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mul┬
%batch_normalization_5/batchnorm/mul_1Muldense_18/Exp:y:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:         2'
%batch_normalization_5/batchnorm/mul_1╙
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2╘
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOp┘
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/sub▌
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:         2'
%batch_normalization_5/batchnorm/add_1й
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_19/MatMul/ReadVariableOp▓
dense_19/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_19/MatMulи
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_19/BiasAdd/ReadVariableOpж
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_19/BiasAddq
dense_19/ExpExpdense_19/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_19/Exp╢
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_6/moments/mean/reduction_indices▄
"batch_normalization_6/moments/meanMeandense_19/Exp:y:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2$
"batch_normalization_6/moments/mean┐
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:	А2,
*batch_normalization_6/moments/StopGradientё
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense_19/Exp:y:03batch_normalization_6/moments/StopGradient:output:0*
T0*(
_output_shapes
:         А21
/batch_normalization_6/moments/SquaredDifference╛
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_6/moments/variance/reduction_indicesЛ
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2(
&batch_normalization_6/moments/variance├
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_6/moments/Squeeze╦
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1▀
+batch_normalization_6/AssignMovingAvg/decayConst*>
_class4
20loc:@batch_normalization_6/AssignMovingAvg/10403*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_6/AssignMovingAvg/decay╒
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_6_assignmovingavg_10403*
_output_shapes	
:А*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp▒
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*>
_class4
20loc:@batch_normalization_6/AssignMovingAvg/10403*
_output_shapes	
:А2+
)batch_normalization_6/AssignMovingAvg/subи
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0*
T0*>
_class4
20loc:@batch_normalization_6/AssignMovingAvg/10403*
_output_shapes	
:А2+
)batch_normalization_6/AssignMovingAvg/mulГ
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_6_assignmovingavg_10403-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_6/AssignMovingAvg/10403*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpх
-batch_normalization_6/AssignMovingAvg_1/decayConst*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg_1/10409*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_6/AssignMovingAvg_1/decay█
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_6_assignmovingavg_1_10409*
_output_shapes	
:А*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp╗
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg_1/10409*
_output_shapes	
:А2-
+batch_normalization_6/AssignMovingAvg_1/sub▓
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg_1/10409*
_output_shapes	
:А2-
+batch_normalization_6/AssignMovingAvg_1/mulП
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_6_assignmovingavg_1_10409/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg_1/10409*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_6/batchnorm/add/y█
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/addж
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/Rsqrtс
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOp▐
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/mul├
%batch_normalization_6/batchnorm/mul_1Muldense_19/Exp:y:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_6/batchnorm/mul_1╘
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/mul_2╒
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_6/batchnorm/ReadVariableOp┌
#batch_normalization_6/batchnorm/subSub6batch_normalization_6/batchnorm/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/sub▐
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_6/batchnorm/add_1й
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02 
dense_20/MatMul/ReadVariableOp▒
dense_20/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_20/MatMulз
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_20/BiasAdd/ReadVariableOpе
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_20/BiasAddp
dense_20/ExpExpdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
dense_20/Exp╢
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_7/moments/mean/reduction_indices█
"batch_normalization_7/moments/meanMeandense_20/Exp:y:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2$
"batch_normalization_7/moments/mean╛
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes

:@2,
*batch_normalization_7/moments/StopGradientЁ
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_20/Exp:y:03batch_normalization_7/moments/StopGradient:output:0*
T0*'
_output_shapes
:         @21
/batch_normalization_7/moments/SquaredDifference╛
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_7/moments/variance/reduction_indicesК
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2(
&batch_normalization_7/moments/variance┬
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_7/moments/Squeeze╩
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1▀
+batch_normalization_7/AssignMovingAvg/decayConst*>
_class4
20loc:@batch_normalization_7/AssignMovingAvg/10442*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_7/AssignMovingAvg/decay╘
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_7_assignmovingavg_10442*
_output_shapes
:@*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOp░
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*>
_class4
20loc:@batch_normalization_7/AssignMovingAvg/10442*
_output_shapes
:@2+
)batch_normalization_7/AssignMovingAvg/subз
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0*
T0*>
_class4
20loc:@batch_normalization_7/AssignMovingAvg/10442*
_output_shapes
:@2+
)batch_normalization_7/AssignMovingAvg/mulГ
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_7_assignmovingavg_10442-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_7/AssignMovingAvg/10442*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpх
-batch_normalization_7/AssignMovingAvg_1/decayConst*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg_1/10448*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_7/AssignMovingAvg_1/decay┌
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_7_assignmovingavg_1_10448*
_output_shapes
:@*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp║
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg_1/10448*
_output_shapes
:@2-
+batch_normalization_7/AssignMovingAvg_1/sub▒
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg_1/10448*
_output_shapes
:@2-
+batch_normalization_7/AssignMovingAvg_1/mulП
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_7_assignmovingavg_1_10448/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg_1/10448*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_7/batchnorm/add/y┌
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_7/batchnorm/addе
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_7/batchnorm/Rsqrtр
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOp▌
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_7/batchnorm/mul┬
%batch_normalization_7/batchnorm/mul_1Muldense_20/Exp:y:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_7/batchnorm/mul_1╙
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_7/batchnorm/mul_2╘
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_7/batchnorm/ReadVariableOp┘
#batch_normalization_7/batchnorm/subSub6batch_normalization_7/batchnorm/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_7/batchnorm/sub▌
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_7/batchnorm/add_1й
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02 
dense_21/MatMul/ReadVariableOp▓
dense_21/MatMulMatMul)batch_normalization_7/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/MatMulи
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_21/BiasAdd/ReadVariableOpж
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/BiasAdd╢
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_8/moments/mean/reduction_indicesх
"batch_normalization_8/moments/meanMeandense_21/BiasAdd:output:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2$
"batch_normalization_8/moments/mean┐
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:	А2,
*batch_normalization_8/moments/StopGradient·
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferencedense_21/BiasAdd:output:03batch_normalization_8/moments/StopGradient:output:0*
T0*(
_output_shapes
:         А21
/batch_normalization_8/moments/SquaredDifference╛
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_8/moments/variance/reduction_indicesЛ
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2(
&batch_normalization_8/moments/variance├
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_8/moments/Squeeze╦
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1▀
+batch_normalization_8/AssignMovingAvg/decayConst*>
_class4
20loc:@batch_normalization_8/AssignMovingAvg/10480*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_8/AssignMovingAvg/decay╒
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_8_assignmovingavg_10480*
_output_shapes	
:А*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOp▒
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*>
_class4
20loc:@batch_normalization_8/AssignMovingAvg/10480*
_output_shapes	
:А2+
)batch_normalization_8/AssignMovingAvg/subи
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:04batch_normalization_8/AssignMovingAvg/decay:output:0*
T0*>
_class4
20loc:@batch_normalization_8/AssignMovingAvg/10480*
_output_shapes	
:А2+
)batch_normalization_8/AssignMovingAvg/mulГ
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_8_assignmovingavg_10480-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_8/AssignMovingAvg/10480*
_output_shapes
 *
dtype02;
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpх
-batch_normalization_8/AssignMovingAvg_1/decayConst*@
_class6
42loc:@batch_normalization_8/AssignMovingAvg_1/10486*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_8/AssignMovingAvg_1/decay█
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_8_assignmovingavg_1_10486*
_output_shapes	
:А*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp╗
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*@
_class6
42loc:@batch_normalization_8/AssignMovingAvg_1/10486*
_output_shapes	
:А2-
+batch_normalization_8/AssignMovingAvg_1/sub▓
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:06batch_normalization_8/AssignMovingAvg_1/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_8/AssignMovingAvg_1/10486*
_output_shapes	
:А2-
+batch_normalization_8/AssignMovingAvg_1/mulП
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_8_assignmovingavg_1_10486/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_8/AssignMovingAvg_1/10486*
_output_shapes
 *
dtype02=
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_8/batchnorm/add/y█
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_8/batchnorm/addж
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_8/batchnorm/Rsqrtс
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_8/batchnorm/mul/ReadVariableOp▐
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_8/batchnorm/mul╠
%batch_normalization_8/batchnorm/mul_1Muldense_21/BiasAdd:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_8/batchnorm/mul_1╘
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_8/batchnorm/mul_2╒
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_8/batchnorm/ReadVariableOp┌
#batch_normalization_8/batchnorm/subSub6batch_normalization_8/batchnorm/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_8/batchnorm/sub▐
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_8/batchnorm/add_1й
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_22/MatMul/ReadVariableOp▒
dense_22/MatMulMatMul)batch_normalization_8/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_22/MatMulз
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_22/BiasAdd/ReadVariableOpе
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_22/BiasAdd|
dense_22/SigmoidSigmoiddense_22/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_22/SigmoidЖ
dense_22/mulMuldense_22/BiasAdd:output:0dense_22/Sigmoid:y:0*
T0*'
_output_shapes
:         2
dense_22/mulv
dense_22/IdentityIdentitydense_22/mul:z:0*
T0*'
_output_shapes
:         2
dense_22/Identity╪
dense_22/IdentityN	IdentityNdense_22/mul:z:0dense_22/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-10510*:
_output_shapes(
&:         :         2
dense_22/IdentityNи
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_23/MatMul/ReadVariableOpг
dense_23/MatMulMatMuldense_22/IdentityN:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpе
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SigmoidSigmoiddense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/SigmoidЖ
dense_23/mulMuldense_23/BiasAdd:output:0dense_23/Sigmoid:y:0*
T0*'
_output_shapes
:         2
dense_23/mulv
dense_23/IdentityIdentitydense_23/mul:z:0*
T0*'
_output_shapes
:         2
dense_23/Identity╪
dense_23/IdentityN	IdentityNdense_23/mul:z:0dense_23/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-10522*:
_output_shapes(
&:         :         2
dense_23/IdentityN╫
IdentityIdentitydense_23/IdentityN:output:0:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_8/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          ::::::::::::::::::::::::::::::2v
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┘
}
(__inference_dense_17_layer_call_fn_10805

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_95882
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╤
л
C__inference_dense_21_layer_call_and_return_conditional_losses_11121

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
█
}
(__inference_dense_21_layer_call_fn_11130

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_21_layer_call_and_return_conditional_losses_98002
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
В
У
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10881

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Р
У
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_11186

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А:::::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Б
Т
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9422

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @:::::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ўС
╙
G__inference_functional_7_layer_call_and_return_conditional_losses_10655

inputs+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource?
;batch_normalization_5_batchnorm_mul_readvariableop_resource=
9batch_normalization_5_batchnorm_readvariableop_1_resource=
9batch_normalization_5_batchnorm_readvariableop_2_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource?
;batch_normalization_6_batchnorm_mul_readvariableop_resource=
9batch_normalization_6_batchnorm_readvariableop_1_resource=
9batch_normalization_6_batchnorm_readvariableop_2_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource;
7batch_normalization_7_batchnorm_readvariableop_resource?
;batch_normalization_7_batchnorm_mul_readvariableop_resource=
9batch_normalization_7_batchnorm_readvariableop_1_resource=
9batch_normalization_7_batchnorm_readvariableop_2_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource;
7batch_normalization_8_batchnorm_readvariableop_resource?
;batch_normalization_8_batchnorm_mul_readvariableop_resource=
9batch_normalization_8_batchnorm_readvariableop_1_resource=
9batch_normalization_8_batchnorm_readvariableop_2_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource
identityИи
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_17/MatMul/ReadVariableOpО
dense_17/MatMulMatMulinputs&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_17/MatMulз
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_17/BiasAdd/ReadVariableOpе
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_17/BiasAddp
dense_17/EluEludense_17/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_17/Eluи
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_18/MatMul/ReadVariableOpв
dense_18/MatMulMatMuldense_17/Elu:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_18/MatMulз
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_18/BiasAdd/ReadVariableOpе
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_18/BiasAddp
dense_18/ExpExpdense_18/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_18/Exp╘
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpУ
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_5/batchnorm/add/yр
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/addе
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtр
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOp▌
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mul┬
%batch_normalization_5/batchnorm/mul_1Muldense_18/Exp:y:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:         2'
%batch_normalization_5/batchnorm/mul_1┌
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1▌
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2┌
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2█
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/sub▌
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:         2'
%batch_normalization_5/batchnorm/add_1й
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_19/MatMul/ReadVariableOp▓
dense_19/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_19/MatMulи
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_19/BiasAdd/ReadVariableOpж
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_19/BiasAddq
dense_19/ExpExpdense_19/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_19/Exp╒
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_6/batchnorm/ReadVariableOpУ
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_6/batchnorm/add/yс
#batch_normalization_6/batchnorm/addAddV26batch_normalization_6/batchnorm/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/addж
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/Rsqrtс
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOp▐
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/mul├
%batch_normalization_6/batchnorm/mul_1Muldense_19/Exp:y:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_6/batchnorm/mul_1█
0batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_1▐
%batch_normalization_6/batchnorm/mul_2Mul8batch_normalization_6/batchnorm/ReadVariableOp_1:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/mul_2█
0batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_2▄
#batch_normalization_6/batchnorm/subSub8batch_normalization_6/batchnorm/ReadVariableOp_2:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/sub▐
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_6/batchnorm/add_1й
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02 
dense_20/MatMul/ReadVariableOp▒
dense_20/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_20/MatMulз
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_20/BiasAdd/ReadVariableOpе
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_20/BiasAddp
dense_20/ExpExpdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
dense_20/Exp╘
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_7/batchnorm/ReadVariableOpУ
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_7/batchnorm/add/yр
#batch_normalization_7/batchnorm/addAddV26batch_normalization_7/batchnorm/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_7/batchnorm/addе
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_7/batchnorm/Rsqrtр
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOp▌
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_7/batchnorm/mul┬
%batch_normalization_7/batchnorm/mul_1Muldense_20/Exp:y:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_7/batchnorm/mul_1┌
0batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_1▌
%batch_normalization_7/batchnorm/mul_2Mul8batch_normalization_7/batchnorm/ReadVariableOp_1:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_7/batchnorm/mul_2┌
0batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_2█
#batch_normalization_7/batchnorm/subSub8batch_normalization_7/batchnorm/ReadVariableOp_2:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_7/batchnorm/sub▌
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_7/batchnorm/add_1й
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02 
dense_21/MatMul/ReadVariableOp▓
dense_21/MatMulMatMul)batch_normalization_7/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/MatMulи
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_21/BiasAdd/ReadVariableOpж
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_21/BiasAdd╒
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_8/batchnorm/ReadVariableOpУ
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_8/batchnorm/add/yс
#batch_normalization_8/batchnorm/addAddV26batch_normalization_8/batchnorm/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_8/batchnorm/addж
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_8/batchnorm/Rsqrtс
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_8/batchnorm/mul/ReadVariableOp▐
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_8/batchnorm/mul╠
%batch_normalization_8/batchnorm/mul_1Muldense_21/BiasAdd:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_8/batchnorm/mul_1█
0batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_8/batchnorm/ReadVariableOp_1▐
%batch_normalization_8/batchnorm/mul_2Mul8batch_normalization_8/batchnorm/ReadVariableOp_1:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_8/batchnorm/mul_2█
0batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_8/batchnorm/ReadVariableOp_2▄
#batch_normalization_8/batchnorm/subSub8batch_normalization_8/batchnorm/ReadVariableOp_2:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_8/batchnorm/sub▐
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2'
%batch_normalization_8/batchnorm/add_1й
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_22/MatMul/ReadVariableOp▒
dense_22/MatMulMatMul)batch_normalization_8/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_22/MatMulз
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_22/BiasAdd/ReadVariableOpе
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_22/BiasAdd|
dense_22/SigmoidSigmoiddense_22/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_22/SigmoidЖ
dense_22/mulMuldense_22/BiasAdd:output:0dense_22/Sigmoid:y:0*
T0*'
_output_shapes
:         2
dense_22/mulv
dense_22/IdentityIdentitydense_22/mul:z:0*
T0*'
_output_shapes
:         2
dense_22/Identity╪
dense_22/IdentityN	IdentityNdense_22/mul:z:0dense_22/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-10636*:
_output_shapes(
&:         :         2
dense_22/IdentityNи
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_23/MatMul/ReadVariableOpг
dense_23/MatMulMatMuldense_22/IdentityN:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpе
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_23/BiasAdd|
dense_23/SigmoidSigmoiddense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_23/SigmoidЖ
dense_23/mulMuldense_23/BiasAdd:output:0dense_23/Sigmoid:y:0*
T0*'
_output_shapes
:         2
dense_23/mulv
dense_23/IdentityIdentitydense_23/mul:z:0*
T0*'
_output_shapes
:         2
dense_23/Identity╪
dense_23/IdentityN	IdentityNdense_23/mul:z:0dense_23/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-10648*:
_output_shapes(
&:         :         2
dense_23/IdentityNo
IdentityIdentitydense_23/IdentityN:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          :::::::::::::::::::::::::::::::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Б
Т
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9142

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Р
У
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10983

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А:::::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ь
к
B__inference_dense_20_layer_call_and_return_conditional_losses_9739

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddU
ExpExpBiasAdd:output:0*
T0*'
_output_shapes
:         @2
Exp[
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Щ)
─
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9249

inputs
assignmovingavg_9224
assignmovingavg_1_9230)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/9224*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9224*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/9224*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/9224*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9224AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/9224*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/9230*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9230*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/9230*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/9230*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9230AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/9230*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1╢
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┤
и
5__inference_batch_normalization_5_layer_call_fn_10907

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_91422
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и)
╟
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_11166

inputs
assignmovingavg_11141
assignmovingavg_1_11147)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Э
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/11141*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11141*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp├
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/11141*
_output_shapes	
:А2
AssignMovingAvg/sub║
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/11141*
_output_shapes	
:А2
AssignMovingAvg/mul 
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11141AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/11141*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpг
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/11147*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11147*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp═
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/11147*
_output_shapes	
:А2
AssignMovingAvg_1/sub─
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/11147*
_output_shapes	
:А2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11147AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/11147*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1╢
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Э
л
C__inference_dense_20_layer_call_and_return_conditional_losses_11020

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddU
ExpExpBiasAdd:output:0*
T0*'
_output_shapes
:         @2
Exp[
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╕А
А
!__inference__traced_restore_11475
file_prefix$
 assignvariableop_dense_17_kernel$
 assignvariableop_1_dense_17_bias&
"assignvariableop_2_dense_18_kernel$
 assignvariableop_3_dense_18_bias2
.assignvariableop_4_batch_normalization_5_gamma1
-assignvariableop_5_batch_normalization_5_beta8
4assignvariableop_6_batch_normalization_5_moving_mean<
8assignvariableop_7_batch_normalization_5_moving_variance&
"assignvariableop_8_dense_19_kernel$
 assignvariableop_9_dense_19_bias3
/assignvariableop_10_batch_normalization_6_gamma2
.assignvariableop_11_batch_normalization_6_beta9
5assignvariableop_12_batch_normalization_6_moving_mean=
9assignvariableop_13_batch_normalization_6_moving_variance'
#assignvariableop_14_dense_20_kernel%
!assignvariableop_15_dense_20_bias3
/assignvariableop_16_batch_normalization_7_gamma2
.assignvariableop_17_batch_normalization_7_beta9
5assignvariableop_18_batch_normalization_7_moving_mean=
9assignvariableop_19_batch_normalization_7_moving_variance'
#assignvariableop_20_dense_21_kernel%
!assignvariableop_21_dense_21_bias3
/assignvariableop_22_batch_normalization_8_gamma2
.assignvariableop_23_batch_normalization_8_beta9
5assignvariableop_24_batch_normalization_8_moving_mean=
9assignvariableop_25_batch_normalization_8_moving_variance'
#assignvariableop_26_dense_22_kernel%
!assignvariableop_27_dense_22_bias'
#assignvariableop_28_dense_23_kernel%
!assignvariableop_29_dense_23_bias
identity_31ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╫
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*у
value┘B╓B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╠
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╟
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Р
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_dense_17_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_17_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2з
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_18_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3е
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_18_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4│
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_5_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5▓
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_5_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6╣
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_5_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7╜
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_5_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8з
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_19_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9е
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_19_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╖
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_6_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╢
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_6_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╜
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_6_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13┴
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_6_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14л
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_20_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15й
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_20_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╖
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_7_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╢
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_7_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╜
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_7_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19┴
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_7_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20л
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_21_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21й
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_21_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╖
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_8_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╢
AssignVariableOp_23AssignVariableOp.assignvariableop_23_batch_normalization_8_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╜
AssignVariableOp_24AssignVariableOp5assignvariableop_24_batch_normalization_8_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25┴
AssignVariableOp_25AssignVariableOp9assignvariableop_25_batch_normalization_8_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26л
AssignVariableOp_26AssignVariableOp#assignvariableop_26_dense_22_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27й
AssignVariableOp_27AssignVariableOp!assignvariableop_27_dense_22_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28л
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_23_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29й
AssignVariableOp_29AssignVariableOp!assignvariableop_29_dense_23_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_299
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЄ
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_30х
Identity_31IdentityIdentity_30:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_31"#
identity_31Identity_31:output:0*Н
_input_shapes|
z: ::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Б)
─
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9389

inputs
assignmovingavg_9364
assignmovingavg_1_9370)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/9364*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayС
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9364*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp┴
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/9364*
_output_shapes
:@2
AssignMovingAvg/sub╕
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/9364*
_output_shapes
:@2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9364AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/9364*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/9370*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЧ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9370*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╦
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/9370*
_output_shapes
:@2
AssignMovingAvg_1/sub┬
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/9370*
_output_shapes
:@2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9370AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/9370*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2
batchnorm/add_1╡
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Я
к
B__inference_dense_19_layer_call_and_return_conditional_losses_9677

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddV
ExpExpBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Exp\
IdentityIdentityExp:y:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
}
(__inference_dense_23_layer_call_fn_11262

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_23_layer_call_and_return_conditional_losses_98992
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
}
(__inference_dense_18_layer_call_fn_10825

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_96152
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
█
}
(__inference_dense_20_layer_call_fn_11029

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_97392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
пD
∙
__inference__traced_save_11375
file_prefix.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_10fda11f5f8547ef82afc72c5818a852/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*у
value┘B╓B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╞
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesю
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ў
_input_shapesх
т: :  : : ::::::	А:А:А:А:А:А:	А@:@:@:@:@:@:	@А:А:А:А:А:А:	А:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%	!

_output_shapes
:	А:!


_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:%!

_output_shapes
:	А@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	@А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
М
╕
,__inference_functional_7_layer_call_fn_10785

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_functional_7_layer_call_and_return_conditional_losses_102092
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
│
н
C__inference_dense_23_layer_call_and_return_conditional_losses_11253

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:         2

Identity┤
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-11246*:
_output_shapes(
&:         :         2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
П
Т
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9282

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А:::::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╨
к
B__inference_dense_21_layer_call_and_return_conditional_losses_9800

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Р)
╟
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_11065

inputs
assignmovingavg_11040
assignmovingavg_1_11046)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Э
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/11040*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11040*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/11040*
_output_shapes
:@2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/11040*
_output_shapes
:@2
AssignMovingAvg/mul 
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11040AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/11040*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpг
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/11046*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11046*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/11046*
_output_shapes
:@2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/11046*
_output_shapes
:@2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11046AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/11046*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2
batchnorm/add_1╡
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┤│
▓
__inference__wrapped_model_9013
input_48
4functional_7_dense_17_matmul_readvariableop_resource9
5functional_7_dense_17_biasadd_readvariableop_resource8
4functional_7_dense_18_matmul_readvariableop_resource9
5functional_7_dense_18_biasadd_readvariableop_resourceH
Dfunctional_7_batch_normalization_5_batchnorm_readvariableop_resourceL
Hfunctional_7_batch_normalization_5_batchnorm_mul_readvariableop_resourceJ
Ffunctional_7_batch_normalization_5_batchnorm_readvariableop_1_resourceJ
Ffunctional_7_batch_normalization_5_batchnorm_readvariableop_2_resource8
4functional_7_dense_19_matmul_readvariableop_resource9
5functional_7_dense_19_biasadd_readvariableop_resourceH
Dfunctional_7_batch_normalization_6_batchnorm_readvariableop_resourceL
Hfunctional_7_batch_normalization_6_batchnorm_mul_readvariableop_resourceJ
Ffunctional_7_batch_normalization_6_batchnorm_readvariableop_1_resourceJ
Ffunctional_7_batch_normalization_6_batchnorm_readvariableop_2_resource8
4functional_7_dense_20_matmul_readvariableop_resource9
5functional_7_dense_20_biasadd_readvariableop_resourceH
Dfunctional_7_batch_normalization_7_batchnorm_readvariableop_resourceL
Hfunctional_7_batch_normalization_7_batchnorm_mul_readvariableop_resourceJ
Ffunctional_7_batch_normalization_7_batchnorm_readvariableop_1_resourceJ
Ffunctional_7_batch_normalization_7_batchnorm_readvariableop_2_resource8
4functional_7_dense_21_matmul_readvariableop_resource9
5functional_7_dense_21_biasadd_readvariableop_resourceH
Dfunctional_7_batch_normalization_8_batchnorm_readvariableop_resourceL
Hfunctional_7_batch_normalization_8_batchnorm_mul_readvariableop_resourceJ
Ffunctional_7_batch_normalization_8_batchnorm_readvariableop_1_resourceJ
Ffunctional_7_batch_normalization_8_batchnorm_readvariableop_2_resource8
4functional_7_dense_22_matmul_readvariableop_resource9
5functional_7_dense_22_biasadd_readvariableop_resource8
4functional_7_dense_23_matmul_readvariableop_resource9
5functional_7_dense_23_biasadd_readvariableop_resource
identityИ╧
+functional_7/dense_17/MatMul/ReadVariableOpReadVariableOp4functional_7_dense_17_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+functional_7/dense_17/MatMul/ReadVariableOp╢
functional_7/dense_17/MatMulMatMulinput_43functional_7/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
functional_7/dense_17/MatMul╬
,functional_7/dense_17/BiasAdd/ReadVariableOpReadVariableOp5functional_7_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,functional_7/dense_17/BiasAdd/ReadVariableOp┘
functional_7/dense_17/BiasAddBiasAdd&functional_7/dense_17/MatMul:product:04functional_7/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
functional_7/dense_17/BiasAddЧ
functional_7/dense_17/EluElu&functional_7/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:          2
functional_7/dense_17/Elu╧
+functional_7/dense_18/MatMul/ReadVariableOpReadVariableOp4functional_7_dense_18_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+functional_7/dense_18/MatMul/ReadVariableOp╓
functional_7/dense_18/MatMulMatMul'functional_7/dense_17/Elu:activations:03functional_7/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
functional_7/dense_18/MatMul╬
,functional_7/dense_18/BiasAdd/ReadVariableOpReadVariableOp5functional_7_dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,functional_7/dense_18/BiasAdd/ReadVariableOp┘
functional_7/dense_18/BiasAddBiasAdd&functional_7/dense_18/MatMul:product:04functional_7/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
functional_7/dense_18/BiasAddЧ
functional_7/dense_18/ExpExp&functional_7/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:         2
functional_7/dense_18/Exp√
;functional_7/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOpDfunctional_7_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;functional_7/batch_normalization_5/batchnorm/ReadVariableOpн
2functional_7/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_7/batch_normalization_5/batchnorm/add/yФ
0functional_7/batch_normalization_5/batchnorm/addAddV2Cfunctional_7/batch_normalization_5/batchnorm/ReadVariableOp:value:0;functional_7/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0functional_7/batch_normalization_5/batchnorm/add╠
2functional_7/batch_normalization_5/batchnorm/RsqrtRsqrt4functional_7/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:24
2functional_7/batch_normalization_5/batchnorm/RsqrtЗ
?functional_7/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_7_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?functional_7/batch_normalization_5/batchnorm/mul/ReadVariableOpС
0functional_7/batch_normalization_5/batchnorm/mulMul6functional_7/batch_normalization_5/batchnorm/Rsqrt:y:0Gfunctional_7/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0functional_7/batch_normalization_5/batchnorm/mulЎ
2functional_7/batch_normalization_5/batchnorm/mul_1Mulfunctional_7/dense_18/Exp:y:04functional_7/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:         24
2functional_7/batch_normalization_5/batchnorm/mul_1Б
=functional_7/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_7_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=functional_7/batch_normalization_5/batchnorm/ReadVariableOp_1С
2functional_7/batch_normalization_5/batchnorm/mul_2MulEfunctional_7/batch_normalization_5/batchnorm/ReadVariableOp_1:value:04functional_7/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2functional_7/batch_normalization_5/batchnorm/mul_2Б
=functional_7/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_7_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02?
=functional_7/batch_normalization_5/batchnorm/ReadVariableOp_2П
0functional_7/batch_normalization_5/batchnorm/subSubEfunctional_7/batch_normalization_5/batchnorm/ReadVariableOp_2:value:06functional_7/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0functional_7/batch_normalization_5/batchnorm/subС
2functional_7/batch_normalization_5/batchnorm/add_1AddV26functional_7/batch_normalization_5/batchnorm/mul_1:z:04functional_7/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:         24
2functional_7/batch_normalization_5/batchnorm/add_1╨
+functional_7/dense_19/MatMul/ReadVariableOpReadVariableOp4functional_7_dense_19_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02-
+functional_7/dense_19/MatMul/ReadVariableOpц
functional_7/dense_19/MatMulMatMul6functional_7/batch_normalization_5/batchnorm/add_1:z:03functional_7/dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
functional_7/dense_19/MatMul╧
,functional_7/dense_19/BiasAdd/ReadVariableOpReadVariableOp5functional_7_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,functional_7/dense_19/BiasAdd/ReadVariableOp┌
functional_7/dense_19/BiasAddBiasAdd&functional_7/dense_19/MatMul:product:04functional_7/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
functional_7/dense_19/BiasAddШ
functional_7/dense_19/ExpExp&functional_7/dense_19/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
functional_7/dense_19/Exp№
;functional_7/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOpDfunctional_7_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;functional_7/batch_normalization_6/batchnorm/ReadVariableOpн
2functional_7/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_7/batch_normalization_6/batchnorm/add/yХ
0functional_7/batch_normalization_6/batchnorm/addAddV2Cfunctional_7/batch_normalization_6/batchnorm/ReadVariableOp:value:0;functional_7/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А22
0functional_7/batch_normalization_6/batchnorm/add═
2functional_7/batch_normalization_6/batchnorm/RsqrtRsqrt4functional_7/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:А24
2functional_7/batch_normalization_6/batchnorm/RsqrtИ
?functional_7/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_7_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02A
?functional_7/batch_normalization_6/batchnorm/mul/ReadVariableOpТ
0functional_7/batch_normalization_6/batchnorm/mulMul6functional_7/batch_normalization_6/batchnorm/Rsqrt:y:0Gfunctional_7/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А22
0functional_7/batch_normalization_6/batchnorm/mulў
2functional_7/batch_normalization_6/batchnorm/mul_1Mulfunctional_7/dense_19/Exp:y:04functional_7/batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А24
2functional_7/batch_normalization_6/batchnorm/mul_1В
=functional_7/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_7_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02?
=functional_7/batch_normalization_6/batchnorm/ReadVariableOp_1Т
2functional_7/batch_normalization_6/batchnorm/mul_2MulEfunctional_7/batch_normalization_6/batchnorm/ReadVariableOp_1:value:04functional_7/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:А24
2functional_7/batch_normalization_6/batchnorm/mul_2В
=functional_7/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_7_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02?
=functional_7/batch_normalization_6/batchnorm/ReadVariableOp_2Р
0functional_7/batch_normalization_6/batchnorm/subSubEfunctional_7/batch_normalization_6/batchnorm/ReadVariableOp_2:value:06functional_7/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А22
0functional_7/batch_normalization_6/batchnorm/subТ
2functional_7/batch_normalization_6/batchnorm/add_1AddV26functional_7/batch_normalization_6/batchnorm/mul_1:z:04functional_7/batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А24
2functional_7/batch_normalization_6/batchnorm/add_1╨
+functional_7/dense_20/MatMul/ReadVariableOpReadVariableOp4functional_7_dense_20_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02-
+functional_7/dense_20/MatMul/ReadVariableOpх
functional_7/dense_20/MatMulMatMul6functional_7/batch_normalization_6/batchnorm/add_1:z:03functional_7/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
functional_7/dense_20/MatMul╬
,functional_7/dense_20/BiasAdd/ReadVariableOpReadVariableOp5functional_7_dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,functional_7/dense_20/BiasAdd/ReadVariableOp┘
functional_7/dense_20/BiasAddBiasAdd&functional_7/dense_20/MatMul:product:04functional_7/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
functional_7/dense_20/BiasAddЧ
functional_7/dense_20/ExpExp&functional_7/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
functional_7/dense_20/Exp√
;functional_7/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOpDfunctional_7_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02=
;functional_7/batch_normalization_7/batchnorm/ReadVariableOpн
2functional_7/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_7/batch_normalization_7/batchnorm/add/yФ
0functional_7/batch_normalization_7/batchnorm/addAddV2Cfunctional_7/batch_normalization_7/batchnorm/ReadVariableOp:value:0;functional_7/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:@22
0functional_7/batch_normalization_7/batchnorm/add╠
2functional_7/batch_normalization_7/batchnorm/RsqrtRsqrt4functional_7/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:@24
2functional_7/batch_normalization_7/batchnorm/RsqrtЗ
?functional_7/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_7_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02A
?functional_7/batch_normalization_7/batchnorm/mul/ReadVariableOpС
0functional_7/batch_normalization_7/batchnorm/mulMul6functional_7/batch_normalization_7/batchnorm/Rsqrt:y:0Gfunctional_7/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@22
0functional_7/batch_normalization_7/batchnorm/mulЎ
2functional_7/batch_normalization_7/batchnorm/mul_1Mulfunctional_7/dense_20/Exp:y:04functional_7/batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @24
2functional_7/batch_normalization_7/batchnorm/mul_1Б
=functional_7/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_7_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02?
=functional_7/batch_normalization_7/batchnorm/ReadVariableOp_1С
2functional_7/batch_normalization_7/batchnorm/mul_2MulEfunctional_7/batch_normalization_7/batchnorm/ReadVariableOp_1:value:04functional_7/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:@24
2functional_7/batch_normalization_7/batchnorm/mul_2Б
=functional_7/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_7_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02?
=functional_7/batch_normalization_7/batchnorm/ReadVariableOp_2П
0functional_7/batch_normalization_7/batchnorm/subSubEfunctional_7/batch_normalization_7/batchnorm/ReadVariableOp_2:value:06functional_7/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@22
0functional_7/batch_normalization_7/batchnorm/subС
2functional_7/batch_normalization_7/batchnorm/add_1AddV26functional_7/batch_normalization_7/batchnorm/mul_1:z:04functional_7/batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @24
2functional_7/batch_normalization_7/batchnorm/add_1╨
+functional_7/dense_21/MatMul/ReadVariableOpReadVariableOp4functional_7_dense_21_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02-
+functional_7/dense_21/MatMul/ReadVariableOpц
functional_7/dense_21/MatMulMatMul6functional_7/batch_normalization_7/batchnorm/add_1:z:03functional_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
functional_7/dense_21/MatMul╧
,functional_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5functional_7_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,functional_7/dense_21/BiasAdd/ReadVariableOp┌
functional_7/dense_21/BiasAddBiasAdd&functional_7/dense_21/MatMul:product:04functional_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
functional_7/dense_21/BiasAdd№
;functional_7/batch_normalization_8/batchnorm/ReadVariableOpReadVariableOpDfunctional_7_batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;functional_7/batch_normalization_8/batchnorm/ReadVariableOpн
2functional_7/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_7/batch_normalization_8/batchnorm/add/yХ
0functional_7/batch_normalization_8/batchnorm/addAddV2Cfunctional_7/batch_normalization_8/batchnorm/ReadVariableOp:value:0;functional_7/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А22
0functional_7/batch_normalization_8/batchnorm/add═
2functional_7/batch_normalization_8/batchnorm/RsqrtRsqrt4functional_7/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:А24
2functional_7/batch_normalization_8/batchnorm/RsqrtИ
?functional_7/batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_7_batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02A
?functional_7/batch_normalization_8/batchnorm/mul/ReadVariableOpТ
0functional_7/batch_normalization_8/batchnorm/mulMul6functional_7/batch_normalization_8/batchnorm/Rsqrt:y:0Gfunctional_7/batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А22
0functional_7/batch_normalization_8/batchnorm/mulА
2functional_7/batch_normalization_8/batchnorm/mul_1Mul&functional_7/dense_21/BiasAdd:output:04functional_7/batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А24
2functional_7/batch_normalization_8/batchnorm/mul_1В
=functional_7/batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_7_batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02?
=functional_7/batch_normalization_8/batchnorm/ReadVariableOp_1Т
2functional_7/batch_normalization_8/batchnorm/mul_2MulEfunctional_7/batch_normalization_8/batchnorm/ReadVariableOp_1:value:04functional_7/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:А24
2functional_7/batch_normalization_8/batchnorm/mul_2В
=functional_7/batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_7_batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02?
=functional_7/batch_normalization_8/batchnorm/ReadVariableOp_2Р
0functional_7/batch_normalization_8/batchnorm/subSubEfunctional_7/batch_normalization_8/batchnorm/ReadVariableOp_2:value:06functional_7/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А22
0functional_7/batch_normalization_8/batchnorm/subТ
2functional_7/batch_normalization_8/batchnorm/add_1AddV26functional_7/batch_normalization_8/batchnorm/mul_1:z:04functional_7/batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А24
2functional_7/batch_normalization_8/batchnorm/add_1╨
+functional_7/dense_22/MatMul/ReadVariableOpReadVariableOp4functional_7_dense_22_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02-
+functional_7/dense_22/MatMul/ReadVariableOpх
functional_7/dense_22/MatMulMatMul6functional_7/batch_normalization_8/batchnorm/add_1:z:03functional_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
functional_7/dense_22/MatMul╬
,functional_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5functional_7_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,functional_7/dense_22/BiasAdd/ReadVariableOp┘
functional_7/dense_22/BiasAddBiasAdd&functional_7/dense_22/MatMul:product:04functional_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
functional_7/dense_22/BiasAddг
functional_7/dense_22/SigmoidSigmoid&functional_7/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:         2
functional_7/dense_22/Sigmoid║
functional_7/dense_22/mulMul&functional_7/dense_22/BiasAdd:output:0!functional_7/dense_22/Sigmoid:y:0*
T0*'
_output_shapes
:         2
functional_7/dense_22/mulЭ
functional_7/dense_22/IdentityIdentityfunctional_7/dense_22/mul:z:0*
T0*'
_output_shapes
:         2 
functional_7/dense_22/IdentityЛ
functional_7/dense_22/IdentityN	IdentityNfunctional_7/dense_22/mul:z:0&functional_7/dense_22/BiasAdd:output:0*
T
2**
_gradient_op_typeCustomGradient-8994*:
_output_shapes(
&:         :         2!
functional_7/dense_22/IdentityN╧
+functional_7/dense_23/MatMul/ReadVariableOpReadVariableOp4functional_7_dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+functional_7/dense_23/MatMul/ReadVariableOp╫
functional_7/dense_23/MatMulMatMul(functional_7/dense_22/IdentityN:output:03functional_7/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
functional_7/dense_23/MatMul╬
,functional_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5functional_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,functional_7/dense_23/BiasAdd/ReadVariableOp┘
functional_7/dense_23/BiasAddBiasAdd&functional_7/dense_23/MatMul:product:04functional_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
functional_7/dense_23/BiasAddг
functional_7/dense_23/SigmoidSigmoid&functional_7/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2
functional_7/dense_23/Sigmoid║
functional_7/dense_23/mulMul&functional_7/dense_23/BiasAdd:output:0!functional_7/dense_23/Sigmoid:y:0*
T0*'
_output_shapes
:         2
functional_7/dense_23/mulЭ
functional_7/dense_23/IdentityIdentityfunctional_7/dense_23/mul:z:0*
T0*'
_output_shapes
:         2 
functional_7/dense_23/IdentityЛ
functional_7/dense_23/IdentityN	IdentityNfunctional_7/dense_23/mul:z:0&functional_7/dense_23/BiasAdd:output:0*
T
2**
_gradient_op_typeCustomGradient-9006*:
_output_shapes(
&:         :         2!
functional_7/dense_23/IdentityN|
IdentityIdentity(functional_7/dense_23/IdentityN:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          :::::::::::::::::::::::::::::::P L
'
_output_shapes
:          
!
_user_specified_name	input_4
Д
╕
,__inference_functional_7_layer_call_fn_10720

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_functional_7_layer_call_and_return_conditional_losses_100692
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╢
и
5__inference_batch_normalization_8_layer_call_fn_11199

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_95292
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
▒
м
B__inference_dense_23_layer_call_and_return_conditional_losses_9899

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:         2

Identity│
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2**
_gradient_op_typeCustomGradient-9892*:
_output_shapes(
&:         :         2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▓
и
5__inference_batch_normalization_7_layer_call_fn_11098

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_93892
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
П
Т
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9562

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А:::::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
и)
╟
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10963

inputs
assignmovingavg_10938
assignmovingavg_1_10944)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Э
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/10938*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10938*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp├
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/10938*
_output_shapes	
:А2
AssignMovingAvg/sub║
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/10938*
_output_shapes	
:А2
AssignMovingAvg/mul 
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10938AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/10938*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpг
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/10944*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10944*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp═
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/10944*
_output_shapes	
:А2
AssignMovingAvg_1/sub─
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/10944*
_output_shapes	
:А2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10944AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/10944*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1╢
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
П
╣
,__inference_functional_7_layer_call_fn_10272
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_functional_7_layer_call_and_return_conditional_losses_102092
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:          
!
_user_specified_name	input_4
Ъ
л
C__inference_dense_18_layer_call_and_return_conditional_losses_10816

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddU
ExpExpBiasAdd:output:0*
T0*'
_output_shapes
:         2
Exp[
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┤
м
B__inference_dense_22_layer_call_and_return_conditional_losses_9867

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:         2

Identity│
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2**
_gradient_op_typeCustomGradient-9860*:
_output_shapes(
&:         :         2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
█A
├

G__inference_functional_7_layer_call_and_return_conditional_losses_10209

inputs
dense_17_10137
dense_17_10139
dense_18_10142
dense_18_10144
batch_normalization_5_10147
batch_normalization_5_10149
batch_normalization_5_10151
batch_normalization_5_10153
dense_19_10156
dense_19_10158
batch_normalization_6_10161
batch_normalization_6_10163
batch_normalization_6_10165
batch_normalization_6_10167
dense_20_10170
dense_20_10172
batch_normalization_7_10175
batch_normalization_7_10177
batch_normalization_7_10179
batch_normalization_7_10181
dense_21_10184
dense_21_10186
batch_normalization_8_10189
batch_normalization_8_10191
batch_normalization_8_10193
batch_normalization_8_10195
dense_22_10198
dense_22_10200
dense_23_10203
dense_23_10205
identityИв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв-batch_normalization_8/StatefulPartitionedCallв dense_17/StatefulPartitionedCallв dense_18/StatefulPartitionedCallв dense_19/StatefulPartitionedCallв dense_20/StatefulPartitionedCallв dense_21/StatefulPartitionedCallв dense_22/StatefulPartitionedCallв dense_23/StatefulPartitionedCallР
 dense_17/StatefulPartitionedCallStatefulPartitionedCallinputsdense_17_10137dense_17_10139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_95882"
 dense_17/StatefulPartitionedCall│
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_10142dense_18_10144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_96152"
 dense_18/StatefulPartitionedCall▓
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_5_10147batch_normalization_5_10149batch_normalization_5_10151batch_normalization_5_10153*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_91422/
-batch_normalization_5/StatefulPartitionedCall┴
 dense_19/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_19_10156dense_19_10158*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_96772"
 dense_19/StatefulPartitionedCall│
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0batch_normalization_6_10161batch_normalization_6_10163batch_normalization_6_10165batch_normalization_6_10167*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92822/
-batch_normalization_6/StatefulPartitionedCall└
 dense_20/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_20_10170dense_20_10172*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_97392"
 dense_20/StatefulPartitionedCall▓
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_7_10175batch_normalization_7_10177batch_normalization_7_10179batch_normalization_7_10181*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_94222/
-batch_normalization_7/StatefulPartitionedCall┴
 dense_21/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_21_10184dense_21_10186*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_21_layer_call_and_return_conditional_losses_98002"
 dense_21/StatefulPartitionedCall│
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_8_10189batch_normalization_8_10191batch_normalization_8_10193batch_normalization_8_10195*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_95622/
-batch_normalization_8/StatefulPartitionedCall└
 dense_22/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_22_10198dense_22_10200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_22_layer_call_and_return_conditional_losses_98672"
 dense_22/StatefulPartitionedCall│
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_10203dense_23_10205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_23_layer_call_and_return_conditional_losses_98992"
 dense_23/StatefulPartitionedCall▓
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          ::::::::::::::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╢
и
5__inference_batch_normalization_6_layer_call_fn_10996

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92492
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
▓
и
5__inference_batch_normalization_5_layer_call_fn_10894

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_91092
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Щ
к
B__inference_dense_18_layer_call_and_return_conditional_losses_9615

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddU
ExpExpBiasAdd:output:0*
T0*'
_output_shapes
:         2
Exp[
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╧A
┴

G__inference_functional_7_layer_call_and_return_conditional_losses_10069

inputs
dense_17_9997
dense_17_9999
dense_18_10002
dense_18_10004
batch_normalization_5_10007
batch_normalization_5_10009
batch_normalization_5_10011
batch_normalization_5_10013
dense_19_10016
dense_19_10018
batch_normalization_6_10021
batch_normalization_6_10023
batch_normalization_6_10025
batch_normalization_6_10027
dense_20_10030
dense_20_10032
batch_normalization_7_10035
batch_normalization_7_10037
batch_normalization_7_10039
batch_normalization_7_10041
dense_21_10044
dense_21_10046
batch_normalization_8_10049
batch_normalization_8_10051
batch_normalization_8_10053
batch_normalization_8_10055
dense_22_10058
dense_22_10060
dense_23_10063
dense_23_10065
identityИв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв-batch_normalization_8/StatefulPartitionedCallв dense_17/StatefulPartitionedCallв dense_18/StatefulPartitionedCallв dense_19/StatefulPartitionedCallв dense_20/StatefulPartitionedCallв dense_21/StatefulPartitionedCallв dense_22/StatefulPartitionedCallв dense_23/StatefulPartitionedCallО
 dense_17/StatefulPartitionedCallStatefulPartitionedCallinputsdense_17_9997dense_17_9999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_95882"
 dense_17/StatefulPartitionedCall│
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_10002dense_18_10004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_96152"
 dense_18/StatefulPartitionedCall░
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_5_10007batch_normalization_5_10009batch_normalization_5_10011batch_normalization_5_10013*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_91092/
-batch_normalization_5/StatefulPartitionedCall┴
 dense_19/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_19_10016dense_19_10018*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_96772"
 dense_19/StatefulPartitionedCall▒
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0batch_normalization_6_10021batch_normalization_6_10023batch_normalization_6_10025batch_normalization_6_10027*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92492/
-batch_normalization_6/StatefulPartitionedCall└
 dense_20/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_20_10030dense_20_10032*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_97392"
 dense_20/StatefulPartitionedCall░
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_7_10035batch_normalization_7_10037batch_normalization_7_10039batch_normalization_7_10041*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_93892/
-batch_normalization_7/StatefulPartitionedCall┴
 dense_21/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_21_10044dense_21_10046*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_21_layer_call_and_return_conditional_losses_98002"
 dense_21/StatefulPartitionedCall▒
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_8_10049batch_normalization_8_10051batch_normalization_8_10053batch_normalization_8_10055*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_95292/
-batch_normalization_8/StatefulPartitionedCall└
 dense_22/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_22_10058dense_22_10060*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_22_layer_call_and_return_conditional_losses_98672"
 dense_22/StatefulPartitionedCall│
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_10063dense_23_10065*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_23_layer_call_and_return_conditional_losses_98992"
 dense_23/StatefulPartitionedCall▓
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          ::::::::::::::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
▐
░
#__inference_signature_wrapper_10339
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__wrapped_model_90132
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:          
!
_user_specified_name	input_4
Р)
╟
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10861

inputs
assignmovingavg_10836
assignmovingavg_1_10842)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Э
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/10836*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10836*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/10836*
_output_shapes
:2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/10836*
_output_shapes
:2
AssignMovingAvg/mul 
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10836AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/10836*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpг
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/10842*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10842*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/10842*
_output_shapes
:2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/10842*
_output_shapes
:2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10842AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/10842*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1╡
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┤
и
5__inference_batch_normalization_7_layer_call_fn_11111

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_94222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╕
и
5__inference_batch_normalization_6_layer_call_fn_11009

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92822
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
З
╣
,__inference_functional_7_layer_call_fn_10132
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_functional_7_layer_call_and_return_conditional_losses_100692
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:          
!
_user_specified_name	input_4
Б)
─
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9109

inputs
assignmovingavg_9084
assignmovingavg_1_9090)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/9084*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayС
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9084*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┴
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/9084*
_output_shapes
:2
AssignMovingAvg/sub╕
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/9084*
_output_shapes
:2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9084AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/9084*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/9090*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЧ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9090*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╦
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/9090*
_output_shapes
:2
AssignMovingAvg_1/sub┬
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/9090*
_output_shapes
:2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9090AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/9090*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1╡
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╢
н
C__inference_dense_22_layer_call_and_return_conditional_losses_11228

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:         2

Identity┤
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-11221*:
_output_shapes(
&:         :         2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
█
}
(__inference_dense_19_layer_call_fn_10927

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_96772
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
бA
е

F__inference_functional_7_layer_call_and_return_conditional_losses_9991
input_4
dense_17_9919
dense_17_9921
dense_18_9924
dense_18_9926
batch_normalization_5_9929
batch_normalization_5_9931
batch_normalization_5_9933
batch_normalization_5_9935
dense_19_9938
dense_19_9940
batch_normalization_6_9943
batch_normalization_6_9945
batch_normalization_6_9947
batch_normalization_6_9949
dense_20_9952
dense_20_9954
batch_normalization_7_9957
batch_normalization_7_9959
batch_normalization_7_9961
batch_normalization_7_9963
dense_21_9966
dense_21_9968
batch_normalization_8_9971
batch_normalization_8_9973
batch_normalization_8_9975
batch_normalization_8_9977
dense_22_9980
dense_22_9982
dense_23_9985
dense_23_9987
identityИв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв-batch_normalization_8/StatefulPartitionedCallв dense_17/StatefulPartitionedCallв dense_18/StatefulPartitionedCallв dense_19/StatefulPartitionedCallв dense_20/StatefulPartitionedCallв dense_21/StatefulPartitionedCallв dense_22/StatefulPartitionedCallв dense_23/StatefulPartitionedCallП
 dense_17/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_17_9919dense_17_9921*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_95882"
 dense_17/StatefulPartitionedCall▒
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_9924dense_18_9926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_96152"
 dense_18/StatefulPartitionedCallо
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_5_9929batch_normalization_5_9931batch_normalization_5_9933batch_normalization_5_9935*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_91422/
-batch_normalization_5/StatefulPartitionedCall┐
 dense_19/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_19_9938dense_19_9940*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_96772"
 dense_19/StatefulPartitionedCallп
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0batch_normalization_6_9943batch_normalization_6_9945batch_normalization_6_9947batch_normalization_6_9949*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92822/
-batch_normalization_6/StatefulPartitionedCall╛
 dense_20/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_20_9952dense_20_9954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_97392"
 dense_20/StatefulPartitionedCallо
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_7_9957batch_normalization_7_9959batch_normalization_7_9961batch_normalization_7_9963*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_94222/
-batch_normalization_7/StatefulPartitionedCall┐
 dense_21/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_21_9966dense_21_9968*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_21_layer_call_and_return_conditional_losses_98002"
 dense_21/StatefulPartitionedCallп
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_8_9971batch_normalization_8_9973batch_normalization_8_9975batch_normalization_8_9977*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_95622/
-batch_normalization_8/StatefulPartitionedCall╛
 dense_22/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_22_9980dense_22_9982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_22_layer_call_and_return_conditional_losses_98672"
 dense_22/StatefulPartitionedCall▒
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_9985dense_23_9987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_23_layer_call_and_return_conditional_losses_98992"
 dense_23/StatefulPartitionedCall▓
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          ::::::::::::::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:P L
'
_output_shapes
:          
!
_user_specified_name	input_4
Щ)
─
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9529

inputs
assignmovingavg_9504
assignmovingavg_1_9510)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/9504*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9504*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/9504*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/9504*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9504AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/9504*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/9510*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9510*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/9510*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/9510*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9510AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/9510*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1╢
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ЩA
е

F__inference_functional_7_layer_call_and_return_conditional_losses_9916
input_4
dense_17_9599
dense_17_9601
dense_18_9626
dense_18_9628
batch_normalization_5_9657
batch_normalization_5_9659
batch_normalization_5_9661
batch_normalization_5_9663
dense_19_9688
dense_19_9690
batch_normalization_6_9719
batch_normalization_6_9721
batch_normalization_6_9723
batch_normalization_6_9725
dense_20_9750
dense_20_9752
batch_normalization_7_9781
batch_normalization_7_9783
batch_normalization_7_9785
batch_normalization_7_9787
dense_21_9811
dense_21_9813
batch_normalization_8_9842
batch_normalization_8_9844
batch_normalization_8_9846
batch_normalization_8_9848
dense_22_9878
dense_22_9880
dense_23_9910
dense_23_9912
identityИв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв-batch_normalization_8/StatefulPartitionedCallв dense_17/StatefulPartitionedCallв dense_18/StatefulPartitionedCallв dense_19/StatefulPartitionedCallв dense_20/StatefulPartitionedCallв dense_21/StatefulPartitionedCallв dense_22/StatefulPartitionedCallв dense_23/StatefulPartitionedCallП
 dense_17/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_17_9599dense_17_9601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_95882"
 dense_17/StatefulPartitionedCall▒
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_9626dense_18_9628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_96152"
 dense_18/StatefulPartitionedCallм
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_5_9657batch_normalization_5_9659batch_normalization_5_9661batch_normalization_5_9663*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_91092/
-batch_normalization_5/StatefulPartitionedCall┐
 dense_19/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_19_9688dense_19_9690*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_96772"
 dense_19/StatefulPartitionedCallн
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0batch_normalization_6_9719batch_normalization_6_9721batch_normalization_6_9723batch_normalization_6_9725*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92492/
-batch_normalization_6/StatefulPartitionedCall╛
 dense_20/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_20_9750dense_20_9752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_97392"
 dense_20/StatefulPartitionedCallм
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_7_9781batch_normalization_7_9783batch_normalization_7_9785batch_normalization_7_9787*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_93892/
-batch_normalization_7/StatefulPartitionedCall┐
 dense_21/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_21_9811dense_21_9813*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_21_layer_call_and_return_conditional_losses_98002"
 dense_21/StatefulPartitionedCallн
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_8_9842batch_normalization_8_9844batch_normalization_8_9846batch_normalization_8_9848*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_95292/
-batch_normalization_8/StatefulPartitionedCall╛
 dense_22/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_22_9878dense_22_9880*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_22_layer_call_and_return_conditional_losses_98672"
 dense_22/StatefulPartitionedCall▒
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_9910dense_23_9912*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_23_layer_call_and_return_conditional_losses_98992"
 dense_23/StatefulPartitionedCall▓
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*а
_input_shapesО
Л:          ::::::::::::::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:P L
'
_output_shapes
:          
!
_user_specified_name	input_4
В
У
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_11085

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @:::::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs"╕L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*л
serving_defaultЧ
;
input_40
serving_default_input_4:0          <
dense_230
StatefulPartitionedCall:0         tensorflow/serving/predict:О 
Чl
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+Ь&call_and_return_all_conditional_losses
Э__call__
Ю_default_save_signature"Ыg
_tf_keras_network f{"class_name": "Functional", "name": "functional_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 4, "activation": "exponential", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 256, "activation": "exponential", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 64, "activation": "exponential", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["dense_20", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["dense_21", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 4, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_22", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 8, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_23", "inbound_nodes": [[["dense_22", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_23", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 4, "activation": "exponential", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 256, "activation": "exponential", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 64, "activation": "exponential", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["dense_20", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["dense_21", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 4, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_22", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 8, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_23", "inbound_nodes": [[["dense_22", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_23", 0, 0]]}}}
ы"ш
_tf_keras_input_layer╚{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
є

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"╠
_tf_keras_layer▓{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
·

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+б&call_and_return_all_conditional_losses
в__call__"╙
_tf_keras_layer╣{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 4, "activation": "exponential", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
▓	
axis
	gamma
 beta
!moving_mean
"moving_variance
#regularization_losses
$trainable_variables
%	variables
&	keras_api
+г&call_and_return_all_conditional_losses
д__call__"▄
_tf_keras_layer┬{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
·

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"╙
_tf_keras_layer╣{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 256, "activation": "exponential", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
╢	
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2regularization_losses
3trainable_variables
4	variables
5	keras_api
+з&call_and_return_all_conditional_losses
и__call__"р
_tf_keras_layer╞{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
¤

6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+й&call_and_return_all_conditional_losses
к__call__"╓
_tf_keras_layer╝{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 64, "activation": "exponential", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
┤	
<axis
	=gamma
>beta
?moving_mean
@moving_variance
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+л&call_and_return_all_conditional_losses
м__call__"▐
_tf_keras_layer─{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ў

Ekernel
Fbias
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
+н&call_and_return_all_conditional_losses
о__call__"╨
_tf_keras_layer╢{"class_name": "Dense", "name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
╢	
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
+п&call_and_return_all_conditional_losses
░__call__"р
_tf_keras_layer╞{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
Ў

Tkernel
Ubias
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+▒&call_and_return_all_conditional_losses
▓__call__"╧
_tf_keras_layer╡{"class_name": "Dense", "name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 4, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
Є

Zkernel
[bias
\regularization_losses
]trainable_variables
^	variables
_	keras_api
+│&call_and_return_all_conditional_losses
┤__call__"╦
_tf_keras_layer▒{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 8, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
 "
trackable_list_wrapper
╞
0
1
2
3
4
 5
'6
(7
.8
/9
610
711
=12
>13
E14
F15
L16
M17
T18
U19
Z20
[21"
trackable_list_wrapper
Ж
0
1
2
3
4
 5
!6
"7
'8
(9
.10
/11
012
113
614
715
=16
>17
?18
@19
E20
F21
L22
M23
N24
O25
T26
U27
Z28
[29"
trackable_list_wrapper
╬
regularization_losses
trainable_variables
`non_trainable_variables
	variables
ametrics
blayer_metrics

clayers
dlayer_regularization_losses
Э__call__
Ю_default_save_signature
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
-
╡serving_default"
signature_map
!:  2dense_17/kernel
: 2dense_17/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
regularization_losses
trainable_variables
enon_trainable_variables
	variables
fmetrics
glayer_metrics

hlayers
ilayer_regularization_losses
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_18/kernel
:2dense_18/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
regularization_losses
trainable_variables
jnon_trainable_variables
	variables
kmetrics
llayer_metrics

mlayers
nlayer_regularization_losses
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
<
0
 1
!2
"3"
trackable_list_wrapper
░
#regularization_losses
$trainable_variables
onon_trainable_variables
%	variables
pmetrics
qlayer_metrics

rlayers
slayer_regularization_losses
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
": 	А2dense_19/kernel
:А2dense_19/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
░
)regularization_losses
*trainable_variables
tnon_trainable_variables
+	variables
umetrics
vlayer_metrics

wlayers
xlayer_regularization_losses
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_6/gamma
):'А2batch_normalization_6/beta
2:0А (2!batch_normalization_6/moving_mean
6:4А (2%batch_normalization_6/moving_variance
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
<
.0
/1
02
13"
trackable_list_wrapper
░
2regularization_losses
3trainable_variables
ynon_trainable_variables
4	variables
zmetrics
{layer_metrics

|layers
}layer_regularization_losses
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
": 	А@2dense_20/kernel
:@2dense_20/bias
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
│
8regularization_losses
9trainable_variables
~non_trainable_variables
:	variables
metrics
Аlayer_metrics
Бlayers
 Вlayer_regularization_losses
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_7/gamma
(:&@2batch_normalization_7/beta
1:/@ (2!batch_normalization_7/moving_mean
5:3@ (2%batch_normalization_7/moving_variance
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
<
=0
>1
?2
@3"
trackable_list_wrapper
╡
Aregularization_losses
Btrainable_variables
Гnon_trainable_variables
C	variables
Дmetrics
Еlayer_metrics
Жlayers
 Зlayer_regularization_losses
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
": 	@А2dense_21/kernel
:А2dense_21/bias
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
╡
Gregularization_losses
Htrainable_variables
Иnon_trainable_variables
I	variables
Йmetrics
Кlayer_metrics
Лlayers
 Мlayer_regularization_losses
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_8/gamma
):'А2batch_normalization_8/beta
2:0А (2!batch_normalization_8/moving_mean
6:4А (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
<
L0
M1
N2
O3"
trackable_list_wrapper
╡
Pregularization_losses
Qtrainable_variables
Нnon_trainable_variables
R	variables
Оmetrics
Пlayer_metrics
Рlayers
 Сlayer_regularization_losses
░__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
": 	А2dense_22/kernel
:2dense_22/bias
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
╡
Vregularization_losses
Wtrainable_variables
Тnon_trainable_variables
X	variables
Уmetrics
Фlayer_metrics
Хlayers
 Цlayer_regularization_losses
▓__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
!:2dense_23/kernel
:2dense_23/bias
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
╡
\regularization_losses
]trainable_variables
Чnon_trainable_variables
^	variables
Шmetrics
Щlayer_metrics
Ъlayers
 Ыlayer_regularization_losses
┤__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
X
!0
"1
02
13
?4
@5
N6
O7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ш2х
G__inference_functional_7_layer_call_and_return_conditional_losses_10529
G__inference_functional_7_layer_call_and_return_conditional_losses_10655
F__inference_functional_7_layer_call_and_return_conditional_losses_9991
F__inference_functional_7_layer_call_and_return_conditional_losses_9916└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
■2√
,__inference_functional_7_layer_call_fn_10720
,__inference_functional_7_layer_call_fn_10272
,__inference_functional_7_layer_call_fn_10785
,__inference_functional_7_layer_call_fn_10132└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▌2┌
__inference__wrapped_model_9013╢
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_4          
э2ъ
C__inference_dense_17_layer_call_and_return_conditional_losses_10796в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_17_layer_call_fn_10805в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_18_layer_call_and_return_conditional_losses_10816в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_18_layer_call_fn_10825в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▐2█
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10881
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10861┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_batch_normalization_5_layer_call_fn_10907
5__inference_batch_normalization_5_layer_call_fn_10894┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_dense_19_layer_call_and_return_conditional_losses_10918в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_19_layer_call_fn_10927в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▐2█
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10963
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10983┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_batch_normalization_6_layer_call_fn_10996
5__inference_batch_normalization_6_layer_call_fn_11009┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_dense_20_layer_call_and_return_conditional_losses_11020в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_20_layer_call_fn_11029в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▐2█
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_11085
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_11065┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_batch_normalization_7_layer_call_fn_11111
5__inference_batch_normalization_7_layer_call_fn_11098┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_dense_21_layer_call_and_return_conditional_losses_11121в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_21_layer_call_fn_11130в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▐2█
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_11166
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_11186┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_batch_normalization_8_layer_call_fn_11199
5__inference_batch_normalization_8_layer_call_fn_11212┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_dense_22_layer_call_and_return_conditional_losses_11228в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_22_layer_call_fn_11237в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_23_layer_call_and_return_conditional_losses_11253в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_23_layer_call_fn_11262в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
2B0
#__inference_signature_wrapper_10339input_4л
__inference__wrapped_model_9013З"! '(1.0/67@=?>EFOLNMTUZ[0в-
&в#
!К
input_4          
к "3к0
.
dense_23"К
dense_23         ╢
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10861b!" 3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ ╢
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10881b"! 3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ О
5__inference_batch_normalization_5_layer_call_fn_10894U!" 3в0
)в&
 К
inputs         
p
к "К         О
5__inference_batch_normalization_5_layer_call_fn_10907U"! 3в0
)в&
 К
inputs         
p 
к "К         ╕
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10963d01./4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ ╕
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10983d1.0/4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ Р
5__inference_batch_normalization_6_layer_call_fn_10996W01./4в1
*в'
!К
inputs         А
p
к "К         АР
5__inference_batch_normalization_6_layer_call_fn_11009W1.0/4в1
*в'
!К
inputs         А
p 
к "К         А╢
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_11065b?@=>3в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ ╢
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_11085b@=?>3в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ О
5__inference_batch_normalization_7_layer_call_fn_11098U?@=>3в0
)в&
 К
inputs         @
p
к "К         @О
5__inference_batch_normalization_7_layer_call_fn_11111U@=?>3в0
)в&
 К
inputs         @
p 
к "К         @╕
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_11166dNOLM4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ ╕
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_11186dOLNM4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ Р
5__inference_batch_normalization_8_layer_call_fn_11199WNOLM4в1
*в'
!К
inputs         А
p
к "К         АР
5__inference_batch_normalization_8_layer_call_fn_11212WOLNM4в1
*в'
!К
inputs         А
p 
к "К         Аг
C__inference_dense_17_layer_call_and_return_conditional_losses_10796\/в,
%в"
 К
inputs          
к "%в"
К
0          
Ъ {
(__inference_dense_17_layer_call_fn_10805O/в,
%в"
 К
inputs          
к "К          г
C__inference_dense_18_layer_call_and_return_conditional_losses_10816\/в,
%в"
 К
inputs          
к "%в"
К
0         
Ъ {
(__inference_dense_18_layer_call_fn_10825O/в,
%в"
 К
inputs          
к "К         д
C__inference_dense_19_layer_call_and_return_conditional_losses_10918]'(/в,
%в"
 К
inputs         
к "&в#
К
0         А
Ъ |
(__inference_dense_19_layer_call_fn_10927P'(/в,
%в"
 К
inputs         
к "К         Ад
C__inference_dense_20_layer_call_and_return_conditional_losses_11020]670в-
&в#
!К
inputs         А
к "%в"
К
0         @
Ъ |
(__inference_dense_20_layer_call_fn_11029P670в-
&в#
!К
inputs         А
к "К         @д
C__inference_dense_21_layer_call_and_return_conditional_losses_11121]EF/в,
%в"
 К
inputs         @
к "&в#
К
0         А
Ъ |
(__inference_dense_21_layer_call_fn_11130PEF/в,
%в"
 К
inputs         @
к "К         Ад
C__inference_dense_22_layer_call_and_return_conditional_losses_11228]TU0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ |
(__inference_dense_22_layer_call_fn_11237PTU0в-
&в#
!К
inputs         А
к "К         г
C__inference_dense_23_layer_call_and_return_conditional_losses_11253\Z[/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ {
(__inference_dense_23_layer_call_fn_11262OZ[/в,
%в"
 К
inputs         
к "К         ╠
G__inference_functional_7_layer_call_and_return_conditional_losses_10529А!" '(01./67?@=>EFNOLMTUZ[7в4
-в*
 К
inputs          
p

 
к "%в"
К
0         
Ъ ╠
G__inference_functional_7_layer_call_and_return_conditional_losses_10655А"! '(1.0/67@=?>EFOLNMTUZ[7в4
-в*
 К
inputs          
p 

 
к "%в"
К
0         
Ъ ╠
F__inference_functional_7_layer_call_and_return_conditional_losses_9916Б!" '(01./67?@=>EFNOLMTUZ[8в5
.в+
!К
input_4          
p

 
к "%в"
К
0         
Ъ ╠
F__inference_functional_7_layer_call_and_return_conditional_losses_9991Б"! '(1.0/67@=?>EFOLNMTUZ[8в5
.в+
!К
input_4          
p 

 
к "%в"
К
0         
Ъ д
,__inference_functional_7_layer_call_fn_10132t!" '(01./67?@=>EFNOLMTUZ[8в5
.в+
!К
input_4          
p

 
к "К         д
,__inference_functional_7_layer_call_fn_10272t"! '(1.0/67@=?>EFOLNMTUZ[8в5
.в+
!К
input_4          
p 

 
к "К         г
,__inference_functional_7_layer_call_fn_10720s!" '(01./67?@=>EFNOLMTUZ[7в4
-в*
 К
inputs          
p

 
к "К         г
,__inference_functional_7_layer_call_fn_10785s"! '(1.0/67@=?>EFOLNMTUZ[7в4
-в*
 К
inputs          
p 

 
к "К         ║
#__inference_signature_wrapper_10339Т"! '(1.0/67@=?>EFOLNMTUZ[;в8
в 
1к.
,
input_4!К
input_4          "3к0
.
dense_23"К
dense_23         