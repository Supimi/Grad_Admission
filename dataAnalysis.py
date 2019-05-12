from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 9
pd.options.display.float_format = '{:.1f}'.format

graduate_admission_df = pd.read_csv("/home/supimi/ML_project/graduate-admissions/Admission_Predict_Ver1.1.csv", sep=",")

# Re-index the data records
graduate_admission_df = graduate_admission_df.reindex(np.random.permutation(graduate_admission_df.index))

prob_of_admission = graduate_admission_df[["Chance_of_Admit"]]
prob_of_admission = np.array(list(dict(prob_of_admission).values())[0])

gre_score = graduate_admission_df[["GRE_Score"]]
gre_score = np.array(list(dict(gre_score).values())[0])

toefel_score = graduate_admission_df[["TOEFL_Score"]]
toefel_score = np.array(list(dict(toefel_score).values())[0])

uni_rating = graduate_admission_df[["University_Rating"]]
uni_rating = np.array(list(dict(uni_rating).values())[0])

sop = graduate_admission_df[["SOP"]]
sop = np.array(list(dict(sop).values())[0])

lor = graduate_admission_df[["LOR"]]
lor = np.array(list(dict(lor).values())[0])

cgpa = graduate_admission_df[["CGPA"]]
cgpa = np.array(list(dict(cgpa).values())[0])

research = graduate_admission_df[["Research"]]
research = np.array(list(dict(research).values())[0])


plt.xlabel("GRE Score")
plt.ylabel("Chance of Admit")
plt.title("GRE Score Vs. Chane of Admit")
plt.scatter(gre_score, prob_of_admission)
plt.savefig("./analysis_graphs/GRE Score Vs. Chane of Admit.png")
plt.show()

plt.xlabel("TOEFL Score")
plt.ylabel("Chance of Admit")
plt.title("TOEFL Score Vs. Chane of Admit")
plt.scatter(toefel_score, prob_of_admission)
plt.savefig("./analysis_graphs/TOEFL Score Vs. Chane of Admit.png")
plt.show()

plt.xlabel("University_rating")
plt.ylabel("Chance of Admit")
plt.title("University_rating Vs. Chane of Admit")
plt.scatter(uni_rating, prob_of_admission)
plt.savefig("./analysis_graphs/University_rating Vs. Chane of Admit.png")
plt.show()

plt.xlabel("SOP")
plt.ylabel("Chance of Admit")
plt.title("SOP Vs. Chane of Admit")
plt.scatter(sop, prob_of_admission)
plt.savefig("./analysis_graphs/SOP Vs. Chane of Admit.png")
plt.show()

plt.xlabel("LOR")
plt.ylabel("Chance of Admit")
plt.title("LOR Vs. Chane of Admit")
plt.scatter(lor, prob_of_admission)
plt.savefig("./analysis_graphs/LOR Vs. Chane of Admit.png")
plt.show()

plt.xlabel("CGPA")
plt.ylabel("Chance of Admit")
plt.title("CGPA Vs. Chane of Admit")
plt.scatter(cgpa, prob_of_admission)
plt.savefig("./analysis_graphs/CGPA Vs. Chane of Admit.png")
plt.show()

plt.xlabel("Research")
plt.ylabel("Chance of Admit")
plt.title("Research Vs. Chane of Admit")
plt.scatter(research, prob_of_admission)
plt.savefig("./analysis_graphs/Research Vs. Chane of Admit.png")
plt.show()

