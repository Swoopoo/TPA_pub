from read_c_from_txt import *
import nnmoirt
import landweberHelper
import matplotlib.pyplot as plt

C_path = 'C:\\Users\\Knechtrupprecht\\Documents\\Git\\TPA\\Daten\\3_Kreise.txt'
S_path = 'C:\\Users\\Knechtrupprecht\\Documents\\Git\\TPA\\Daten\\S_matrix_q_neu_richtig.mat'
C = readCfromTXT(C_path)
S_imp = landweberHelper.read_matlab_struct(S_path)
S = S_imp['S_matrix_q_neu_richtig']
func = nnmoirt.NNMOIRT(S, [91, 91])

Gfut = func.calc_G(C, residuum=1e-6)
plt.imshow(Gfut.reshape(91, 91))
plt.show()

