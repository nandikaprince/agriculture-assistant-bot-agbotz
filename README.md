TRANSFORMER

Self Attention: Finding the important word in a sentence
input/prompt → model which is trained on a large dataset → output/response
Self attention parameters -> QKV : Query Key Value (all vector values)

Self Attention Softmax = (Qk^T/sqrt(dk))V
dk: dimensionality of matrix

Q-> Is the query/question asked (focusing part)
K-> Is the reply (The features)
V-> In depth features (vectors)

SoftMax = e^n/sigma(e^n)               (we get the probability distribution)
CNN last layer is a SoftMax function.

ARCHITECTURE

All the components are separate blocks which work together connecting one output to the other input to finally give the output of the transformer

Input embedding -—->Positional Embedding —--> Self Attention -—->Feed forward —> Normalization —->Output

1. Input embedding : Convert words to numbers
2. Positional encoding : Adding a new value to the converted number
3. Self-Attention : gives a new set of numbers.
4. Feed-Forward : in-depth feature extraction
5. Normalization : Error reduction.
6. Output 


1) Input embedding

Each word into vectors
Splits the individual words into different components, ie tokenization

Eg: input:-The quick brown fox jumps over the lazy dog
Each word is converted into vectors using embedding.Here 9 words are converted (input
encoding)

Choose dimensionality
Represented by a 4 dimensional representation

2) Positional Encoding

Find position
Word positions are encoded in the sentence
PEeven= sin(pos/(1000^(2i/dmodel)
PEodd= cos(pos/(1000^(2i/dmodel)

Pos: position of the word






3) Self Attention

To capture dependencies and relationships within input sequences.
Used to find the relationship between different elements of the input sequence and to find in depth features of the input sequence

Attention scores are calculated by taking the dot product of a query with the keys.

https://medium.com/@averma9838/self-attention-mechanism-transformers-41d1afea46cf




4) Feed Forward

In-depth feature extraction takes place
Relu fn - It is a non-linear activation function

FFN=max(0,xW1+b1) W2+b2

W1: input to hidden space
W2: project to high dimensional space

5) Normalization

Prove solutions for errors and other issues that might occur with the model
Eg: Gradient explosion, errors etc

To reduce loss, gradient explosion problems
Values from normalization is applied to softmax. This is called token selection.

(x-mu)/sigma * (gamma/beta)

->Training
Means you are inputting your data to a pre existing model and then make the model do well on your input data

->Fine Tuning
Eg: The quick brown fox jumps over the lazy dog.

Layer 1 (Input Encoding) : It tokenizes the sentence and places it as tokens.
Converts each word into a 4d matrix
Layer 2 (Positional Encoding) :
PE(pos,2i)= sin(pos/10,000 2i/dmodel)
PE(pos,2i+1) = cos(pos/10,000 2i/dmodel)

Layer 3 (Self Attention) : Comparing itself(Each word) with other words to get more features.
Self attention = Softmax(QKT / sqrt(dk)) V
Q=XWQ
K=XWK
V=XWV
For example, we are querying on “quick”, K compares the relevance of the words with each other and how much importance can be given to the query. In this particular example, it shows how much the input is relevant to the key.
V is the value vector which gives the aggregation score, i.e. it is an aggregator vector which is highly dependent on Q. We obtain it during training and it represents the in-depth and detailed features of the data.

Layer 4 (Feed forward) :  Max(0,x)    	i.e. RELU function
FFN= max (0,xW1+ b1) W2 + b2
It has MLP layers, i.e. multilayer perceptron layers.
W : weight matrices
b : biases
FFN gives the high dimensional representation which is very hard to visualize. This layer is responsible for artificial generative AI, AI with higher intelligence than humans etc etc

Layer 5 (Normalization) : mapping the outputs to sensible values in the output domain.
Layer normalization(x) = ()  +
X: input(h)
mu: mean of input
: standard deviation
Gamma beta are learned parameters

Layer 6 (Multihead attention) :  Part of the decoder architecture. After this , we perform FFN, normalization etc
Multihead(Q,K,V) = concat(head1 , head2 ….. headi )
headi = attention(Q WiQ , K WiK , V WiV )
These heads are used to perform more feature extraction
Wi  are the weight matrices, multiplied with the query , key and value matrices to perform many attention steps.
 
Layer 7 (Feedforward layer) :
      	Gate —> Up and down
      	Gate → High dimensional projection
      	Up → Deeper high dimensional projection
      	Down → Back to embedded projection
 
Layer 8 (Normalization)
Layer 9 (Output)

