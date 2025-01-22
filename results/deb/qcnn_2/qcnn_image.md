Quantum convolutional neural network for image classification# SHORT PAPER 

Quantum convolutional neural network for image classification 

# Guoming Chen1  · Qiang Chen1 · Shun Long2 · Weiheng Zhu2 · Zeduo Yuan2 · Yilin Wu1 

Received: 30 January 2021 / Accepted: 27 August 2022 / Published online: 24 September 2022 

© The Author(s), under exclusive licence to Springer-Verlag London Ltd., part of Springer Nature 2022 

# Abstract 

In this paper we propose two scale-inspired local feature extraction methods based on Quantum Convolutional Neural Net- work (QCNN) in the Tensorflow quantum framework for binary image classification. The image data is properly downscaled with Multi-scale Entanglement Renormalization Ansatz and Box-counting based fractal features before fed into the QCNN’s quantum circuits for state preparation, quantum convolution and quantum pooling. Quantum classifiers with one QCNN and two hybrid Quantum-classical QCNN models have been trained with a breast cancer dataset, and their performance are compared against that of a classic CNN. The results show that the proposed QCNN with the proposed feature extraction methods outperformed the classic CNN in terms of recognition accuracy. It is interesting to find that image bit-plane slicing has a similar internal mechanism to that of the Ising phase transition. This observation motivates us to explore the correlation between the chaotic nature of image and the classification performance enhancement by QCNN classifiers. It also implies that the pixels of the image and the Ising chaology particles share some similar patterns and are apt to classification. 

Keywords Quantum convolutional neural network · MERA circuit · Image classification · Box-counting · Tensorflow quantum 

# 1 Introduction 

Quantum techniques have brought new inspiration to many traditional machine learning algorithms such as supervised learning, principal component analysis and other dimension reduction algorithms, and have been a hot topic in research in recent years [1–4]. In addition, the performance of many 

* • Guoming Chen  isscdz@mail.sysu.edu.cn  Qiang Chen  cq\_c@gdei.edu.cn 

 Shun Long  tlongshun@jnu.edu.cn 

 Weiheng Zhu  tzhuwh@jnu.edu.cn 

 Zeduo Yuan  yuanzd@stu2020.jnu.edu.cn 

 Yilin Wu  lyw@gdei.edu.cn 

1 School of Computer Science, Guangdong University 

2 Department of Computer Science, Jinan University, of Education, Guangzhou 510303, Guangdong, China Guangzhou 510632, Guangdong, China traditional algorithms such as sparse matrix inversion, data fitting, and low-rank matrix decomposition can match that of quantum phase estimation algorithm. Tang et al. [5] suggested that, despite no evidence about the exponential speedup from quantum machine learning, it is feasible for a low rank approximation of matrices in QRAM where QRAM can be considered as a classical data structure, and the data is systematically measured in the computational basis. Speedups can be achieved via singular value decom- position for those QRAM type samples. This suggests that QRAM plays a vital role by representing vectors as states for both quantum and classic approaches in measurement of the output states. Based on a sparsity assumption of input data, if a classical method can find an appropriate subspace of a significantly lower dimension, quantum machine learn- ing can then be applied on low-rank data. By doing so, the polynomial gaps between classical and quantum computa- tion can be bridged via numerical simulation.

The use of sparsity can be traced back to the breakthrough of neuroscience. A sparse code reflects the simple-cell receptive field character of the natural images and effec- tively solve the problem of signal sparsity in a similar man- ner as the wavelet transform. This led to the emergence of comprehensive sensing widely used in image understanding and classification. In search for a good feature, a large or even over-complete dictionary is essential and likely avail- able in practice via wavelet transforms, which leads to a sparse solution. Generally, a tradeoff between rank and dic- tionary (i.e. sparsity and redundancy) must be made with a concern of the problem (for instance image interpretation and classification). 

Cong [6] proposed a quantum circuit-based algorithm inspired by CNN which makes O(log(N)) variational parameters as input sizes of N qubits. The training results on near-term quantum devices show that QCNNs can accu- rately recognize quantum states corresponding to a symme- try-protected topological phase. Deep neural networks use tensor to reduce redundancy and improve CNN efficiency. Some works focus on using different tensors to parameterize each layer. Kossaifi [7] parameterize the whole CNN with a single higher-order tensor. Correlation between different tensor dimensions is then unveiled to capture the overall net- work structure. In addition, by imposing a low rank structure on the tensor, this parameterization can implicitly regular- ize the whole network, significantly reduce the number of parameters, and achieve higher accuracy and compression. Henderson [8] proposed a quantum convolution layer on input data by locally transforming the data by the deploy- ment of a number of random quantum circuits for local data transformation in order to produce meaningful features for image classification purpose. Broughton et al. [9] proposed a tensorflow quantum framework to simulate quantum cir- cuits. The architecture and compose module can be used for supervised learning in classification. Bravyi et al. [10] proposed a method to classically simulate quantum circuits with non-Clifford gates. The computational cost was found related to the stabilizer rank. Farhi et al. [11] proposed a quantum approximate optimization algorithm which alter- nates use a cost-function base on Hamiltonian and a mix- ing Hamiltonian. The variational quantum eigensolver is a hybrid quantum-classical algorithm to find the eigenvalues of a huge-sized matrix. It has the potential to turn quantum computing in practice in the near future. 

Tensor network can represent sets of correlated data in machine learning. For instance, in quantum many-body sys- tems, it encodes the coefficients of the state wave function and ensembles of microstates which provide a nice expres- sion of multi-dimensional data and has the advantage of dimensionality reduction, i.e. high compression ratio for structured data with low reconstruction error. Tensor net- works promise a natural structural characterization of data, especially with correlations and variable scale. Tensor networks [12–14] can be used for states classification(for example, a MPS, TTN, MERA or PEPS network [15]) and simulate entangled correlated systems. 

1 3 

The contribution of this paper are as follows: (1) We pro- pose two feature extraction methods: MERA and Box-count- ing based fractal features based on Quantum Convolutional Neural Network(QCNN) for binary image classification. 

* (2) 
We analyze two distilled scale features to appropriately com- bine with quantum convolutional neural network to analyze whether the classification pattern of the physical state/phase can be transferred and applied into the studying of traditional image classification issue.
* (3) QCNN classifiers outperform the classic CNN in breast cancer images.
* (4) We find that image bit-plane slicing has a similar internal mechanism to that of the Ising phase transition. We also explore the correla- tion between the chaotic nature of image and the classification performance enhancement by QCNN classifiers.
# 2 Local construct features 

The idea of dividing matter into particles, molecules, and atoms is introduced into the image space. We explore the schemes to measure the scale of particles with degrees of granulation. These local granularity information can be dis- tilled and encoded into a global quantum network. 

## 2.1 MERA image feature extraction 

Prior research found that there is a relationship between dis- crete wavelet transformations and Multi-scale Entanglement Renormalization Ansatz (MERA). But few pay attention to this physical phenomenon to be associated with Multi-scale analysis for extracting local features, especially the operation of truncation is similar to image compression. 

In MERA quantum circuits, unitary gates with reflection symmetry are multi-scale representation of quantum many- body wave function and MERA [16–19] can encode correla- tions with different scales for data compression. Equation 1 is a 2 × 2 unitary matrix denoted as Usw under some reflections. 

[ 

] 

0 1 

* (1)Usw = 1 0 

Equation 2 is a 3 × 3 unitary matrix with one parameter of reflection symmetric matrices v (). 

⎛ 

⎡ 

0 0 

⎤ 

⎞
* (2)v() = exp ⎜⎜⎝ 

1√ 2 

⎢⎢⎣ 

−0 −  

0 0 

⎥⎥⎦ 

⎟⎟⎠ 

It is equivalent to
* (3)v() = 1 

2 

⎡⎢⎢⎢⎣ 

cos() + 1

√ 

2 sin() cos() − 1 

− 

√ 

2 sin() 2 cos() − 

√ 

2 sin() 

cos() − 1 

√ 

2 sin() cos() + 1 

⎤ 

⎥⎥⎥⎦ 

Fig. 1  Two scale schemes of local construct features 

| 

### THI 

D 

B 

C 

d == 

= 

HE) 

= 

log4 log 16 

log log8 log log8 32 32 = Solson 

= } 

=> 

log13 log4 

A 

### B C HASE | 2 

D 

v(0;) -> 

v(8,) 

v(83) - 

| 

| X 

* (a) Scale Layer Unitary Circuits (b) Box-counting based fractal features

The unitary circuit is formed by 3 × 3 reflection symmetric 

matrices v () with the swap gate Usw . It is used to param- 

eterize a symmetric transforms by dilation factor three. The 

transformation of Fig. 1a is a ternary unitary circuit with three rotation angles  as given below. 

V is a 3N × 3N matrix with parameter N in a ternary uni- 

tary circuit, the decomposition form is, 

* (4)V = V3UswV2UswV1

Vk is direct sum of N matrices v(k)from Eq. 3, 

⨁ ⨁ ⨁ 

* (5)Vk = v(k) v(k) v(k) v(k)…

The ternary unitary circuit has parameters 1 , 2 , 3 . Three 

coefficient sequences labelled L, C and R which are entered 

into the ternary circuit. They are called right, left of an edge- 

centered, site-centered symmetric sequence respectively. 

The multi-scale circuit which encodes the images and its 

output are then be choosen to yield the ten output features. 

The process is similar to features extraction from the wavelet 

transformation of the given image by means, variance and 

kurtosises of horizontal,vertical and diagonal subbands. This 

is the first scheme by which we distill features from an image 

and to be integrated into the quantum circuit. 

## 2.2 Fractal scale feature extraction 

In the second scheme, spatial occupancy is a scale to 

measure the pixel intensity distribution in image space. It 

describes irregular degrees at different scale. Fractal dimen- sion is a tool to express image scale. N() is the amount of particles with size  which is defined as: 

* (6)N() ∼ −D

where D is the Box-counting fractal dimension [20–23], as 

defined in Eq. 7 and L is the box size. N(L)is the amount of boxes comprising at least one mass in it. 

logN(L) 

### (7)D = lim 

L→0 

log(1∕L) 

whereas, the Box-counting has a disadvantage in that it 

neglects the amount of masses inside a box Ni(L) . It is mean- ingful to characterize complex spatial structure by estimat- 

ing the mass probability in the ith box as: 

* (8)Pi(L) = Ni(L)∕NT where Ni(L) is the amount of pixels with mass in the ith box 

and NT is the amount of the total mass. Figure 1b depicts 

images with the same histogram differentiate each other at 

different scales. Consider the scale = 1∕L changes (e.g. 

1/2,1/4,1/8,...), where L is the box size. There exists a ratio 

by applying the logarithmic function on that between the 

amount of pixels with mass in the ith box and that of the cor- 

responding scale. With  changes, we get the different ratio 

as Box-counting fractal features. For example, in Fig. 1b, 

= 1∕2 , A can be distinguished from B, C, D. = 1∕4 , A, 

B can be distinguished from C, D. 

With these two scale schemes, information can be dis- 

tilled. How to prepare the classical data for a quantum circuit is by no means trivial. Further research is need to reveal 

its internal mechanisms in the black box quantum network 

where particles and image pixels share some similar pattern. 

3 Quantum convolutional neural network 

Quantum convolutional neural network (QCNN) can be designed to recognize quantum states. It is crucial to study how local features integrate into global QCNN circuit 

Fig. 2  State prepare circuit 

[ @ 

(0, 0) : 

H 

(0, 1): 

H (0, 2): 

H 

(0, 3): 

H 

(0, 4): 


	+ (0. 5): (0, 6) : 
	
	H 
	
	H | 
	
	H 
	
	(0, 7): 
	
	H 
	
	(0, 8): 
	
	H 
	
	(0, 9): 
	
	Hstructure, and how to bridge the gap and make their connec- 

tion. Different from the previous work and recent advances, 

we compare the information distil ability between two scale 

patterns as local correlation features to be integrated into 

global QCNNs and spread into the entire unitary evolution 

system. Next, we will employ QCNNs to analyze whether 

the classification mode of the physical state/phase can be 

transferred into learning the traditional image classification 

problem and check which kind of image is apt to learn. The 

preparation of quantum initial states is important. The higher 

the entangled state is, the higher the separated weight func- 

tion is. It is more convincing that with entangled state. The 

QCNN would have more expressive power than its classical counterpart. 

Rx() ≡ e 2 

I − i sin 

X 

[ 

2 

] 

2 

−i 

X = cos
* (9)U��0⟩ = U 2n−1� 

i�i⟩ = 

2n−1� 

i�i⟩ 

i=0 

i=0 

= 

2 

2 

−i sin 

2 

cos 

2 

cos 

− i sin makes the best of layer classes for quantum circuit construc- 

tion. The first step is to define the quantum circuits, then prop- erly prepare a state and finally train the quantum classifier to detect if it works. “Entanglement” is to accelerate the process- ing. When entanglement is reduced, we can obtain the clas- sification result by reading a single qubit. 

The generality of QCNN architecture for this image classifi- cation task is illustrated in Fig. 5. The first layer of the QCNN architecture is quantum cluster state prepare layer as shown in Fig. 2. Where H gate is applied to any of its qubits indicates an excitation and CZ gate is applied to any of the two adjacent qubits to get the desired state with highly entangled.
* (10)
* (11)
* (12)

Ry() ≡ e 

−i 2 Y = cos  

I − i sin  

Y 

2 

2 

= 

⎡ 

⎢ 

cos  − sin  

2 

⎤ 

2 

⎥ 

⎥ 

⎦ 

⎢ 

⎣ 

sin  

2 cos  

2 

Rz() ≡ e 

−i 2 

Z = cos 

I − i sin 

Z 

[ 

2 

] 

2 

e−i∕2 0 

0 ei∕2 

= 

For a quantum system consider an initial state ��0⟩ = 

∑2n−1 

i=0 

i�i⟩, ∈ ℂ where {i⟩, i = 0, 2,… , 2n − 1} denotes a set of bases in the Hilbert space,i, i ∈ ℂ . The 

QCNN applies the unitary transformation U on it. Quantum 

circuit operates quantum bits form by quantum logic gates 

where quantum logic circuit gate is a linear combination of unitary matrics, meaning that the whole quantum circuit is also a large unitary matrix. However, a problem is how to 

find an optimal parameter setting for the quantum circuit to 

act as an activation function in the network. So it can distills 

the information from local features according to various data distributions. Classic computers can be used to optimize the 

parameters to be fed into the quantum circuit diagram. 

The following steps are about the procedure on how to 

assemble circuits in a Tensorflow. Tensorflow Quantum (TFQ) 

The second layer is the input layer where the encoded fea- 

tures via the two schemes in previous chapters are distilled 

as the rotation angles  of single-qubit RX, RY, RZ gates 

and the rotation angles  of the two qubit XX, YY, ZZ gates 

1 3 

| 

1): 

10.2: 

* (3): .

(0.7: As: MA: 

| xx 

|zx 

| 

| 

| 2'xl4 

| xx0 

|xx | Yx4 

| 

|xxii | 

|2°xi4 \_T}\_r}\_{2) 

| 2] |zy 

- 

234 

xx0 | zx 

| z 

Axel 

| 

E 

22" 

Fig. 3  Quantum convolution 

as inputs to parameterized unitary circuit where XX is sup- 

posed to be tensor product of X with X with rotation angles 

 . Equations 10–12 define RX, RY, RZ gates in the circuit, 

the parameters are regarded as an input to the input layer, 

and they decide the rotation angle around the X, Y and Z 

axis in Bloch sphere. The gradient of the QCNN is rela- 

tively gentle, so local correlation features are important to 

represent data distribution and ultimately affect the gradient. 

It is meaningful to explore the relationship between scale 

inspired dimensionality reduction features and the gradient. 

Then the previous part of quantum circuit are constructed 

by cluster state prepare layer and input layer. 

The convolution and pooling layer include one and two 

qubit parameterized unitary matrices. The third layer is 

quantum convolution layer. Figure 3 depicts RX,RY,RZ and 

XX,YY,ZZ gates in quantum convolution layer that can be 

constructed by a cascade of two-qubit parameterized uni- 

tary to pairs of adjacent qubits gradually. The fourth layer 

is quantum pooling layer. Figure 4 depicts RX,RY,RZ and CNOT gates in quantum pooling layer. CNOT gates are used to control entanglement. Two arbitrary qubit uni- 

tary make a parameterized pooling in the circuit where entanglement is reduced down from two qubits to one qubit unitary circuit. The quantum pooling layer pools half of the qubits using two-qubit pool mentioned above. The pooling layer chooses to output the important qubits 

where the label 1 assigned one state while −1 assigned the 

other state. The pooling layer is followed by the repeated 

measurement observable Z on state �⟩ which is denoted as 

⟨Z⟩�⟩ ≡ ⟨�Z�⟩ = ��2 − ��2 where ⟨Z⟩ ∈ [−1, 1] . In this 

architecture, the classical data is from an image data set for 

binary classification. Pixel is not a suitable image feature 

conducive to put into quantum network for classification, 

in the previous section, we have proposed MERA inspired 

ternary unitary circuit and Box-counting fractal features 

to downscale the image and prepare the ten features as the 

input parameters. 

(0, 0): 

(0, 1): 

* (0. 2): 
	+ (0. 3): . 
		- (0. 4): |- 
			* ro. E: 
				+ (0. 7): (0, 8): . 
				
				
					- (0. 9): \_[x30]}

[x0] 

@ 

xx3 

|Yx | zxs 

x3 

[v] [2>) 

O 

X 

Z7-x2) | 

[x°(20) 

| Yai 

X 

[27s) || X°(30) 

X"x0 | vxi | z22 

xx [2] (2] 

@ 

X 

|262) | ¥<a}[x(20) 

| xx0 rul[2 X 

Z"(x2) || r(x1) 

X°(-x0) 

X"x0 ra \_|zx 

Z"(-x2) | [x(80) 

Fig. 4  Quantum pooling 

Figure 5a illustrates a QCNN architecture constructed by 

above layers. Figure 5b depicts the hybrid QCNN model 

which combines a classical neural network with a single 

quantum convolution and pooling layer. Figure 5c depicts 

the hybrid QCNN with multiple quantum which combines 

multiple quantum convolutions and pooling layer with a 

classical neural network. 

# 4 Experimental results 

This section aims to demonstrate the efficiency of the pro- 

posed QCNN classifier based on scale inspired image fea- 

tures. To this end, we implement two sets of experiment. 

The experiments were carried out in an environment of 

tensorflow-quantum 0.3.0 and cirq 0.8.0. 

The first experiment includes a medical dataset of breast 

cancer images, including 50 images of normal without dis- 

ease, 67 images of benign fibrous tumors and 205 images 

of malignant breast cancer. We divided the dataset into two 

groups(67 benign fibrous tumors and 205 malignant cancer, 

50 normal and 205 malignant cancer). In the first group, 

60% of 272 images were randomly selected for training 

and the remaining 40% for validation. In our experiments, 

the MERA features and Box-counting fractal features were 

normalized to [-− ,  ] as parameters of rotation angles in 

RX, RY, RZ gates and XX, YY, ZZ gates and then spread into the 10 qubits parameterized quantum circuits. This 

1 3 

integration of encoded local correlation features and QCNN 

better highlight the multi-scale nature of data distribution. 

Comparison on performance between the QCNN model 

and the classical CNN model suggests that improvement 

have been achieved via the adoption of quantum features. 

The val loss and val accuracy in Fig. 6a show that QCNN, 

Hybrid QCNN and Hybrid QCNN with multiple quantum layers have achieved accuracies of about 78–93%, and the difference is negligible. A simple classical CNN has 

State 

Prepare 

Circuit 

State 

Prepare 

Circuit 

Input 

layer 

Quantum 

convolution 

Quantum 

Pooling 

* (a) OCNN Input 

layer 

Quantum 

convolution 

Quantum 

Pooling 

Classical 

neural 

network
* (b) Hybrid QCNN Quantum 

convolutio 

Quantum 

Pooling 

State 

Prepar 

e 

Input 

layer 

Quantum 

convolutio Quantum 

Pooling 

Classical 

neural 

network 

Quantum 

convolutio 

Quantum 

Pooling
* (c) Hybrid QCNN with multiple quantum Fig. 5  QCNN and hybrid QCNNs with classical architecture 

0.9 

0.8 

Accuracy 0.7 

Validation 0.6 

0.5 

0.9 


	+ § 0.7 Accur 
	
	Validation 
	
	& 
	
	0.5 
	
	Quantum vs Hybrid CNN performance 
	
	Quantum vs Hybrid CNN performance 
	
	Quantum vs Hybrid CNN performance 
	
	WMING 

classicald CNN 


	+ - classical loss AA 
	
	vin0.9 

0.8 

0.4 

0.3 

O. 

Accuracy 0.7 

Validation 0.6 0.5 


	+ 1.00.9 

0.8 

0.6 

0.4 

0.3 

0.2 

Accuracy 0.7 

Validation 0.5 

Accurac P 

ation 

0 

0.3 

WANT 

NAS 

Epochs 

0 

20 

40 

60 

80 

100 

O 

60 

BO 

O 

20 

40 

60 

80 

100 

A 


	+ - OCNN
	+ - Hybrid CNN
	+ - Hybrid CNN loss
	+ - Hybrid CNN Multi-Quantum QCNN loss
	+ - Multiple Quantum loss 
		- - QCNN
		- - QCNN loss
		- - Hybrid CNN
		- - Hybrid CNN loss
		- - Hybrid CNN Multi-Quantum
		- - Multiple Quantum lossM 


	+ - OCNN
	+ - OCNN loss Hybrid CNN
	+ - Hybrid CNN loss
	+ - Hybrid CNN Multi-Quantum
	+ - Multiple Quantum loss
	+ - OCNN
	+ - OCNN loss Hybrid CNN 
	
	Hybrid CNN loss 
	
	Hybrid CNN Multi-Quantum 
	
	Multiple Quantum loss 
	
	gEpochs 


	+ (c) QCNN(Box) In the First Group 
	
	Quantum vs Hybrid CNN performance 
	
	Quantum vs Hybrid CNN performance 
	
	Quantum vs Hybrid CNN performance 
	
	
		- (a) QCNN(MERA) In the
		- (b) CNN In the First First Group 
		
		Group 
		
		Cary 
		
		-| CNN 

cal loss 

0 

20 

40 

60 

80 

100 

A 

0 

20 

40 

60 

80 

100 

Epochs
* (d) QCNN(MERA) In the Second Group
* (e) CNN In the Second
* (f) QCNN(Box) In the Group 

Second Group 

QCNNs vs CNN performance 

OCNNs vs CNN performance 


	+ 1.0recision/Recall 0.6 0. 

0.2 

0.0 


	+ - OCNN recall
	+ - QCNN precision
	+ - Hybrid CNN recall
	+ - Hybrid CNN precision
	+ - Multiple Quantum recall
	+ - Multiple Quantum precision
	+ - CNN recall
	+ - CNN precision0.4 

ision/Recall 0.6 

0.2 

0.0 


	+ - QCNN recall
	+ - OCNN precision Hybrid CNN recall
	+ - Hybrid CNN precision
	+ - Multiple Quantum recall
	+ - Multiple Quantum precision
	+ - CNN recall
	+ - CNN precision0 

20 

40 

60 

80 

100 

O 

20 

40 

60 

80 

100 

Epochs ochs
* (g) Precition/Recall In
* (h) Precition/Recall In

the First Group 

the Second Group 

Fig. 6  QCNN VS CNN image recognition accuracy 

achieved an accuracy of about 78–85% as shown in Fig. 6b. 

for all three quantum-based models, particularly the pure 

Convergence of pure QCNN has a large fluctuation in a small QCNN one. In Fig. 6d, the images in second group includes 

range when compared with the other two quantum classical 

50 images of normal without disease and 205 images of 

hybrid models. Better convergence may still be expected malignant breast cancer. 60% of these images were also randomly selected for training data and the remaining 40% 

for validation. QCNN, Hybrid QCNN, and Hybrid QCNN 

with multiple quantum layers have achieved accuracies of 

about 83–88%, but vibration and noise have been controlled 

to a certain extent. A simple classical CNN has achieved 

an accuracy of about 82–84% as shown in Fig. 6e in the 

second group. Figure 7a shows some images for accurate 

classification. Figure 7b shows some images for inaccurate 

classification. Figure 6g, h show that the proposed QCNN 

with the proposed feature extraction method outperformed 

the classic CNN in terms of Precisions/Recall in both groups 

of images. And the performance is getting better gradually. 

Similar to a multi-scale image analysis and representa- 

tion method, MERA and fractal features have similar local 

correlations that are significant to study the characteristics 

of images at various scales. Through multi-scale decomposi- 

tion, the image information is distilled in the global network 

structure, which lead to rapid improvement of the QCNN 

performance (e.g. fast increase the accuracy accompanied 

by fast decrease of the loss). Comparisons have been made 

between the MERA and Box-counting fractal features in 

the same QCNN model and each shows a certain advantage 

respectively. Figure 6a, c depicts val accuracy and val loss 

in QCNN, Hybrid QCNN and Hybrid QCNN with multi- 

ple quantum layers on MERA versus Box-counting frac- 

tal features in the 1st category. The accuracies of fractal 

outperforms MERA 78–93% versus 78–98% while MERA 

outperforms fractal in terms of convergence. Figure 6d, f 

depict val accuracy and val loss in QCNN models with com- 

parisons between MERA and fractal in the 2nd category. The accuracies of MERA and fractal are 83–88% versus 

78–98%, we get similar results. 

In statistical physics, Ising model [24–26] can describe the phase transition of ferromagnetic materials. When heated over some temperature threshold, the system loses its mag- netism temporarily until cooled down to that threshold. The transition between magnetic and non-magnetic phases is called phase transition. It is said that ferromagnetic materi- als are composed of many small magnetic needles, each of which has only two orientation directions in Ising model. The neighbor needles interact with each other by some 

energy constraints. Meanwhile, they experience random magnetic transformation from external sources. The degree of reversals depends on certain parameters, e.g. temperature. When temperature arises, the small magnetic needles are in disorder and the magnetism disappears. Once the tempera- ture turns low, the system is in a state of high energy, and a large number of small needles are in the same orientation, and the system shows strong magnetism. When the system is at a critical temperature, Ising model shows a self-similar phenomena. Monte Carlo method and Ising model based Metropolis algorithm are used to generate images which is shown in Fig. 8b. Distribution granularity information of the data can be used in classification, and the result is affected by differ- ent position distribution of the pixel values. Figure 8a depicts that the image bit-plane decomposition has a similar internal mechanism to that of the Ising model where phase transfers from disordered to ordered state. This motivates us to use the chaotic hypotheses of the image data to empirically explore the correlation between the chaotic 

norma 

norma 

benign brous tumors 

benign brous tumors 

malignant cancer 

## (a) Accurate Classification (b) Inaccurate Classification 

malignant cancer 

Fig. 7  Some images for accurate and inaccurate classification 

1 3 

Guy | O 

| 

| 

Lie 

|A 

| 

f 

Z4 

q k 

## O AS 

F 

X | | 

### (a) Bit-planes Slicing (b) Multi-scale Ising Chaology Image 

Fig. 8  The chaotic nature of image 

nature of image and the classification performance through 

bit-plane slicing and gives an interpretation of why it achieve 

better performance than the original one. The experimental 

results show that QCNNs have achieved accuracies of about 

83–98% on bit-plane 1 and bit-plane 2 as shown in Fig. 9a, b, 

they have achieved accuracies of about 84–97% on bit-plane 

3, 83–97% on bit-plane 4, 86–97% on bit-plane 5, 82–96% 

on bit-plane 6, 73–95% on bit-plane 7, 72–94% on bit-plane 

8 as shown in Fig. 9c–h, respectively. Especially bit-plane 2 

and bit-plane 5 have better convergence. It is interesting to 

find that the unstructured or disordered bit-planes outper- 

form the structured or ordered bit-plane. The phase transi- 

tion occurs approximately at bit-plane 5. It achieves both 

the high accuracy and good convergence. We examine the 

impact of phase transition on classification performance 

assessment and conclude that the disordered bit-planes in 

image can leads to further performance improvement. 

In order to investigate the importance of each part of our 

proposed method, we carried out ablation study and reported 

the results. Our ablation study is composed of three parts. 

First, we applied bit-plane slicing on the image and kept 

other blocks unchanged. We performed experiment on the 

bit-plane 5 where the phase transition occurs approximately 

at this bit-plane. We listed the experiment results in Fig. 10a. 

It shows that classification performance in bit-plane 5 is 

better than the original image. Second, we removed the 

pooling layer from the proposed network and kept other 

blocks unchanged. Figure 10b depicts that classification performance has dropped to a certain extent. Third, we removed the convolution layer from the proposed network and kept other blocks unchanged. Figure 10c depicts that classification performance has dropped significantly. Experi- ment results obtained in our ablation study show that each part of our proposed method is indispensable. It suggests that the convolution layer is more important than the pool- ing layer. 

Performance improvement can be expected by fine-tuning the models and their parameters. Our interesting observation suggests that the chaotic nature of the original image will be conducive to the improvement of classification performance. Especially, this phase transition is more likely to be broadly in accordance with the chaotic nature of the image to take into account. 

In the second set of experiments, we associated the 

image pixels with quantum particles. In order to explore the expressive ability of neural networks. For simulation pur- pose, we first used Monte Carlo method and Ising model based Metropolis algorithm to generate 1000 images with 

100 × 100 pixels which represent ten different scale levels 

of chaos complexity(level of pixel confusion). Some of 

the resulting quantum-chaotic Ising images with different 

scales are shown in Fig. 8b. For these 1000 quantum cha- 

otic Ising images, a pre-defined number of clustering has 

been followed in order to fix the number of categories to 

* 10. The [0–0.1], (0.1–0.2], (0.2–0.3], (0.3–0.4], (0.4–0.5],

(0.5–0.6], (0.6–0.7], (0.7–0.8], (0.8–0.9], (0.9–1] region 

1 3 

* 1.0

0.8 

0.6 

0.4 

0.2 

ccuracy 

Validation 

0.8 

Accuracy 0.6 

Validation 0.4 

0.2 

U 

* - QCNN
* - OCNN loss
* - Hybrid CNN
* - Hybrid CNN loss
* - Hybrid CNN Multi-Quantum
* - Multiple Quantum loss

Validation 0.4 

Accuracy 0.6 

Accuracy 0.6 

Validation 0.4 

0.2 

Accuracy 0.6 

Validation 0.4 

0.2 

## WAS 

Epochs 

Quantum vs Hybrid CNN performance 

006 

O 

20 

40 

60 

100 

0 

40 

60 

80 

100 

0 

20 

40 

60 

80 

100 

Epochs 

* (a) bit-plane 1 (b) bit-plane 2 (c) bit-plane 3

Epochs 

Quantum vs Hybrid CNN performance 

Quantum vs Hybrid CNN performance 

* 1.0
Movies 

party won 


	+ - QCNN
	+ - Hybrid CNN loss
	+ - OCNN loss
	+ - Hybrid CNN
	+ - Hybrid CNN Multi-Quantum
	+ - Multiple Quantum loss 
		- - OCNN
		- - OCNN loss
		- - Hybrid CNN
		- - Hybrid CNN loss
		- - Hybrid CNN Multi-Quantum
		- - Multiple Quantum loss
	+ - OCNN
	+ - OCNN loss Hybrid CNN
	+ - Hybrid CNN loss
	+ - Hybrid CNN Multi-Quantum
	+ - Multiple Quantum loss 
		- - OCNN
		- - OCNN loss
		- - Hybrid CNN
		- - Hybrid CNN loss
		- - Hybrid CNN Multi-Quantum
		- - Multiple Quantum loss

0.8 

0.2 

Accuracy 0.6 

Validation 0.4 

* - QCNN
* - OCNN loss
* - Hybrid CNN
* - Hybrid CNN loss
* - Hybrid CNN Multi-Quantum
* - Multiple Quantum loss

Quantum vs Hybrid CNN performance 

Quantum vs 

Hybrid CNN performance 

Quantum vs Hybrid CNN performance 

## WALLAS 

Epochs 

(d) bit-plane 4 (e) bit-plane 5 (f) bit-plane 6 

Epochs 

Epochs 

O 20 

40 

60 

80 

100 

0 

20 

40 

60 

80 

100 

20 

40 

60 

80 

100 

* 1.0

0.8 

Accuracy 0.6 

Validation 0.4 

Quantum vs Hybrid CNN performance 

Quantum vs Hybrid CNN performance 

MAY V 

Vw 

.0 

0.9 

0.8 

0.6 

0.4 0.3 

Accuracy 0.7 

Validation 0.5 

* - OCNN
* - Hybrid CNN loss
* - OCNN loss Hybrid CNN
* - Hybrid CNN Multi-Quantum
* - Multiple Quantum loss

O 

* - OCNN
* - OCNN loss
* - Hybrid CNN
* - Hybrid CNN loss
* - Hybrid CNN Multi-Quantum
* - Multiple Quantum loss

0.2 

## WOHA 

## WADISMAWSAA 

O 20 

40 | 

60 

80 

100 

O 20 

40 

60 

80 

100 

Epochs 

* (g) bit-plane 7 (h) bit-plane 8

Epochs 

Fig. 9  QCNNs classification on image bit-plane slicing 

have been labelled with 1–10, each corresponding to a cat- data distribution for classification. We divided 10 different 

egory representing one of the ten different chaotic levels scale levels of chaology image into five groups [0–0.1] and 

from top to bottom in Fig. 11a, i.e. from fine-grained to (0.9–1], (0.1–0.2] and (0.8–0.9], (0.2–0.3] and (0.7–0.8], 

coarse-grained quantum chaotic images. The granularity (0.3–0.4] and (0.6–0.7], (0.4–0.5] and (0.5–0.6], and carried 

of the division on the chaology image can be considered out binary classification respectively. Figure 11b shows that, 

on information granules. Neural networks built on vari- QCNN, Hybrid QCNN and Hybrid QCNN with multiple 

ous information granules can make use of granularity and quantum layers have achieved accuracies of about 60–97% 

Quantum vs Hybrid CNN performance 

A 

l.0 

0.9 

0.8 

0.6 

0.4 

0.3 

0.2 

Accuracy 0.7 

Validation 0.5 

Validation 0.4 

Accuracy 0.6 

0.2 

| 0 

* - OCNN
* - OCNN loss
* - Hybrid CNN
* - Hybrid CNN loss
* - Hybrid CNN Multi-Quantum
* - Multiple Quantum loss Epochs

60 

80 

100 

O 

20 

40 

60 

80 

100 

O 

20 

40 

60 

80 

100 

Epochs 

Epochs 

* - OCNN
* - OCNN loss
* - Hybrid CNN
* - Hybrid CNN loss
* - Hybrid CNN Multi-Quantum
* - Multiple Quantum loss
* - OCNN
* - OCNN loss
* - Hybrid CNN
* - Hybrid CNN loss
* - Hybrid CNN Multi-Quantum
* - Multiple Quantum loss
## X COMMAN 

Quantum vs Hybrid CNN performance 

Quantum vs Hybrid CNN performance 

## WALLAS 

O 20 40 

O 

0.8 

0.2 

Accuracy 0.6 

Validation 0.4 

### (a) Decompose bit-planes (b) Remove Pooling (c) Remove Convolution 

Fig. 10  Ablation study in our proposed method 

Quantum vs Hybrid CNN performance 

## ENINT 

Quantum vs Hybrid CNN performance 

## VANY 

## RES 

QCNIN loss 

* - Hybrid CNN
* - Hybrid CNN loss
* - Hybrid CNN Multi-Quantur
* - Multiple Quantum loss
## SUMAS 

| 60 

| vs Hybrid CNN ner 

## VE WWW 

### (a) Ten different scale lev- 

els from top to bottom 

### (b) First group (c) Second Group 

F's 

OF ? 

AVIST 

Comic 

d CNN 1 

Hybrid CNN Multi-Quantu 

## W | WWWWNYM 

- 

0.6 

|- 

* - Hybrid CF Hybrid CNN Multi-Quan
* - Hybrid CNN loss Hybrid CNN Multi -Quantum
* - Multiple Quantum loss
* - Hybrid CNN loss
* - Hybrid CNN Multi-Quantur
* - Multiple Quantum loss
## WSP 

N 

V 

20 

20 

60 

80 

60 

80 

| 100 

(d) Third group (e) Fourth group (f) Fifth group 

Fig. 11  Binary classification on five groups of different scale Ising chaology images on groups [0–0.1] and (0.9–1]. Figure 11c shows that accu- racies of about 70–97% have been achieved on groups (0.1–0.2]and (0.8–0.9], but QCNN performance is slightly poorer than those of the other two methods. Figure 11d shows that accuracies of about 74–95% have been achieved on groups (0.2–0.3] and (0.7–0.8] with improvement on con- vergence still expected. Figure 11e shows that accuracies of about 65–97% have been achieved on groups (0.3–0.4] and (0.6–0.7], Fig. 11f shows accuracies of about 75–97% have been achieved on groups (0.3–0.4] and (0.6–0.7], but QCNN performance is slightly weaker than the others’. 

In order to explore the relation between the quantization of pixels of the image and quantum particles, further experi- ment and in-depth analysis have to be carried on to investi- gate scale features in quantum critical system and machine learning in phases of matter. For example, the classifica- tion of paramagneticphase and ferromagneticphase. In the simulation of correlated system, the truncated features show minor loss of accuracy. 

There are some improvements to be made in future. First, the type and number of samples in present research is rela- tively small. In the future, we will collect more different types of images from the real world. Second, more methods of feature extraction based on tensor network needs to be carried out to explore the interpretability and performance improvement of QCNN models. 

# 5 Conclusions 

In this paper we propose two scale-inspired local feature extraction methods for binary pattern breast cancer image classification based on Quantum Convolutional Neural Network(QCNN) in tensorflow quantum framework. The image data is properly downscaled with MERA and Box- counting based fractal features before fed into the quan- tum circuit of QCNN architecture which includes the state preparation, quantum convolution, and quantum pooling. High or appropriate entangled state corresponds to the high separated weight function when entanglement is reduced, and we can obtain the classification result from the qubit. We have trained the Quantum classifiers with QCNN and two types of hybrid quantum-classical QCNN models and compared their results against that of a classic CNN for per- formance evaluation. The simulation results on the breast cancer image datasets show that performance improvement in terms of recognition accuracy and classification accuracy can be achieved by proposed QCNN with the proposed fea- ture extraction methods when compared against the classic CNN. It is interesting to find that image bit-plane slicing has a similar internal mechanism to that of the Ising phase transition. This observation motivates us to explore the 

Acknowledgements The authors would like to acknowledge the financial support from National Key R &D Program of China (2019YFC0120102), Natural Science Foundation of Guangdong Province (Nos. 2018A0303130169, 2022A1515010485), National Natural Science Foundation of China (No. 61772140), the Special Projects in Key Fields of Universities in Guangdong Province (Nos. 2020ZDZX1023, 2021ZDZX1062), and the Opening Project of Guang- dong Province Key Laboratory of Big Data Analysis and Processing at the Sun Yat-sen University(No. 201902). 

# References 

* 1. Biamonte J, Wittek P, Pancotti N (2017) Quantum machine learn- ing. Nature 549(7671):195–202
* 2. Havlcek V, Crcoles AD (2019) Supervised learning with quantum- enhanced spaces. Nature 567(7747):209–212
* 3. Harrow AW, Montanaro A (2017) Quantum computational supremacy. Nature 549(7671):203–209
* 4. Beer K, Bondarenko D, Farrelly T (2020) Training deep quantum neural networks. Nat Commun 11(1):1–6
* 5. Tang E (2019) A quantum-inspired classical algorithm for rec- ommendation systems. In: Proceedings of the 51st annual ACM SIGACT symposium on theory of computing, pp 217–228
* 6. Cong I, Choi S, Lukin MD (2019) Quantum convolutional neural networks. Nat Phys 15(12):1273–1278
* 7. Kossaifi J, Bulat A, Tzimiropoulos G (2019) T-net: parametriz- ing fully convolutional nets with a single high-order tensor. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp 7822–7831
* 8. Henderson M, Shakya S, Pradhan S (2020) Quanvolutional neu- ral networks: powering image recognition with quantum circuits. Quant Mach Intell 2(1):1–9
* 9. Broughton M et al (2020) Tensorflow quantum: a software frame- work for quantum machine learning. Preprint arXiv:​2003.​02989
* 10. Bravyi S, Browne D, Calpin P (2019) Simulation of quantum cir- cuits by low-rank stabilizer decompositions. Quantum 3:181
* 11. Farhi E, Goldstone J, Gutmann S (2014) A quantum approximate optimization algorithm. Preprint arXiv:​1411.​4028
* 12. Yang S, Wang M, Feng Z, Liu Z, Li R (2018) Deep sparse tensor filtering network for synthetic aperture radar images classification. IEEE Trans Neural Netw Learn Syst 29(8):3919–3924
* 13. Stoudenmire E, Schwab DJ (2016) Supervised learning with ten- sor networks. In: Advances in neural information processing sys- tems, pp 4799–4807
* 14. Sun ZZ, Peng C, Liu D, Ran SJ, Su G (2020) Generative tensor network classification model for supervised machine learning. Phys Rev B 101(7):075135
* 15. http://tensornetwork.org/
* 16. Evenbly G, White SR (2016) Entanglement renormalization and wavelets. Phys Rev Lett 116(14):140403
* 17. Haegeman J, Swingle B, Walter M, Cotler J, Evenbly G (2018) Rigorous free-fermion entanglement renormalization from wave- let theory. Phys Rev X 8(1):011003
* 18. Evenbly G, Vidal G (2013) Quantum criticality with the multi- scale entanglement renormalization ansatz. In: Strongly correlated systems. Springer, Berlin, pp 99–130
* 19. Cincio L, Dziarmaga J, Rams MM (2008) Multiscale entangle- ment renormalization ansatz in two dimensions: quantum Ising model. Phys Rev Lett 100(24):240603
* 20. Panigrahy C, Seal A, Mahato NK (2020) Fractal dimension of synthesized and natural color images in lab space. Pattern Anal Appl 23(2):819–836
* 21. Panigrahy C, Seal A, Mahato NK (2021) A new technique for estimating fractal dimension of color images. In: Proceedings of international conference on frontiers in computing and systems. Springer, Singapore, pp 257–265
* 22. Panigrahy C, Seal A, Mahato NK (2019) Is box-height really a issue in differential box counting based fractal dimension? In: 2019 international conference on information technology (ICIT). IEEE, pp 376–381
* 23. Panigrahy C, Seal A, Mahato NK (2020) Image texture surface analysis using an improved differential box counting based fractal dimension. Powder Technol 364:276–299
* 24. Gavrilov A, Jordache A, Vasdani M, Deng J (2018) Convolutional neural networks: estimating relations in the ising model on over- fitting. In: 2018 IEEE 17th international conference on cognitive informatics and cognitive computing. IEEE, pp 154–158
* 25. Kyle M, Isaac T (2018) Deep neural networks for direct, feature- less learning through observation: the case of two-dimensional spin models. Phys Rev E 97(3):032119
* 26. Coyle B, Mills D, Danos V, Kashe E (2020) The Born supremacy: quantum advantage and training of an Ising Born machine. npj Quant Inf 6(1):1–11

Publisher's Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations. 

Springer Nature or its licensor holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law. 

Guoming Chen received the M.S. and Ph.D. degree from School of Information Science and Technology, Sun Yat-sen University, China, in 2003 and 2009. He has been awarded a scholarship under the State Scholarship Fund to pursue his study in Eastern Washington University, USA as a joint Ph.D. student for 12 months in 2008. He is currently an associate professor in Guangdong University of Education. His areas of interest include data mining, machine learning, pattern recognition and image processing. 

Qiang Chen received his B.S. degree from Mathematics Department, South China Normal University, Guangzhou, in 1984. He is a Professor of Guangdong University of Education, China. His current research interests are artificial intelligence, machine learning and knowledge engineering. 

Shun Long received his Bachelor’s from Jinan University in 1995, and his Ph.D. degree from the University of Edinburgh in 2004. He is cur- rently an associate professor in Jinan University. His research interests include machine learning and cognitive science. 

Weiheng Zhu received his Bachelor’s, Master’s and Ph.D. degree from Sun Yat-sen University in 1998, 2003 and 2006 respectively. He is currently a senior lecturer in Jinan University. His research interests include artificial intelligence, machine learning and database. 

Zeduo Yuan received the degree in software engineering from the Guangdong University of education in 2020. He is currently a post- graduate student in Computer Technology at the Jinan University of Guangzhou, China. His research interests are in the areas of artificial intelligence (machine learning and deep learning), medical images processing, multimodal learning. 

Yilin Wu received the Ph.D. degree in automatic control from the South China University of Technology, Guangzhou, China, in 2016. He was an Engineer with Dongfang Boiler Group Company, Ltd., Zigong, China, from 1992 to 2000. He is currently a Professor in Department of Computer Science, Guangdong University of Education. His research interests include complex systems modeling, networked control sys- tems, vibration control and machine vision. 

1 3 

