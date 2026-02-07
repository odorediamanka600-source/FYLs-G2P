from g2p import *        

text = "Since there was no present like the present, he decided to present the project to the lead singer, who was wearing a lead-colored suit in the desert, even though his friends might desert him."
g = G2P()
print("original text:", text)
print("result:", " ".join(g(text)))
print("oov words:", g.preran_oov)