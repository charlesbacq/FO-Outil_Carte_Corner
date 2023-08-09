#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all the necessaries librairies
import pulp as p
import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials


# In[2]:


# import solver
solver = p.GLPK_CMD(path='/Users/charlesbacquaert/anaconda3/envs/Projet_Corners_Outil_Carte/bin/glpsol')

#solver = p.GLPK()
print(solver.available())

solver_list = p.listSolvers(onlyAvailable=True)
print(solver_list)


# In[3]:


# connect to google sheet with the json key
gc = gspread.service_account(filename='projetcorneroutilcarte-8eff842e6082.json')

# connect Sheets of EPD available and constraints
doc_outil_carte = gc.open_by_url("https://docs.google.com/spreadsheets/d/1HmvRWM9CZvMWwRNSt1RftssvzA1aGLVyHSOrvCqHw4c/")
sheet_outil_carte = doc_outil_carte.worksheet("Input")
sheet_contraintes_jours = doc_outil_carte.worksheet("Contraintes Jours")
sheet_contraintes_sous_categories = doc_outil_carte.worksheet("Contraintes Sous Categories")
sheet_contraintes_particulieres = doc_outil_carte.worksheet("Contraintes Particulieres")
sheet_contraintes_parametres = doc_outil_carte.worksheet("Contraintes Paramètres")


# In[4]:


# import constants
constant_M = sheet_contraintes_parametres.acell('D2').value
constant_N = sheet_contraintes_parametres.acell('D4').value

M = int(constant_M)
N = int(constant_N)

print("M is", M)
print("N is", N)

# create constants 
EPD = ["ENTREE","PLAT","DESSERT"]
JOURS = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"] 


# In[5]:


# import input data 
data = sheet_outil_carte.get('A2:AB')
headers = data.pop(0)

# transform the data from google sheet to data frame that Python can read and display the first rows of data
df = pd.DataFrame(data, columns=headers)
print(df.columns)
#print(df.head(5))


# In[6]:


# Define input data

# useful, data in Input
df["ID"].replace('', np.nan, inplace=True)
df.dropna(subset=["ID"], inplace=True)
df["ID"] = df["ID"].astype(int)
df["Fournisseur"]=df["Fournisseur"].astype(str)
df["EPD"]=df["EPD"].astype(str)
df["Sous Categorie Corner"]=df["Sous Categorie Corner"].astype(str)
df["DLC"] = df["DLC"].astype(int)
df["R"] = df["R"].astype(int)
df["FC"] = df["FC"].astype(float)

# useless, data in Input 

#df["PP semaine"]=df["PP semaine"].astype(str) #str au lieu de float 
#df["PP week-end"]=df["PP week-end"].astype(str) #str au lieu de float
#df["AA"] = df["AA"].astype(int)
#df[" Prix"] = df[" Prix"].astype(str) # str au lieu de float
#df["AA lundi"] = df["AA lundi"].astype(int)
#df["Province"] = df["Province"].astype(int)
#df["Régime"] = df["Régime"].astype(str)
#df["Sous Categ Proteine"] = df["Sous Categ Proteine"].astype(str)
#df["Forme légumes"] = df["Forme légumes"].astype(str)
#df["Soupe equivalent"] = df["Soupe equivalent"].astype(str)
#df["Base"] = df["Base"].astype(str)
#df["Chaud/Froid"] = df["Chaud/Froid"].astype(str)
#df["Chocolat"] = df["Chocolat"].astype(str)
#df["F1"] = df["F1"].astype(int)
#df["F2"] = df["F2"].astype(int)
#df["F3"] = df["F3"].astype(int)
#df["Carte Dimanche"] = df["Carte Dimanche"].astype(int)
#df["Fromage blanc_yaourt"] = df["Fromage blanc_yaourt"].astype(int)

# useless, no data in Input
#df["R Province"] = df["R Province"].astype(int)


# In[7]:


# Define the optimization problem
prob = p.LpProblem("Outil_Carte", p.LpMinimize)

# Define the decision variables
carte_vars = p.LpVariable.dicts("Carte", [(product, index_jour) for product in df.index for index_jour in range(len(JOURS))], cat="Binary")
dish_count_vars = p.LpVariable.dicts("Dish_Count", df.index, cat="Integer", lowBound=0)

# Define the binary varimathaables y_i
y_vars = p.LpVariable.dicts("y", df.index, cat=p.LpBinary)


# In[8]:


# Define the objecive expression
penalty_expr = p.lpSum(M*(dish_count_vars[product] - y_vars[product]) for product in df.index)
R_4sem_expr = p.lpSum(N*(int(df.loc[product, "R"]) * carte_vars[(product, index_jour)]) for product in df.index for index_jour in range(len(JOURS)))
food_cost_expr = - p.lpSum(carte_vars[(product, index_jour)] for product in df.index for index_jour in range(len(JOURS)))

objective_expr = food_cost_expr #+ penalty_expr + R_4sem_expr
prob += objective_expr


# In[9]:


# Constraint to update dish count variables
for i in df.index:
    prob += dish_count_vars[i] == p.lpSum(carte_vars[(i, j)] for j in range(7))   
    
# Constraint to enforce y_vars based on dish_count_vars
for i in df.index:
    prob += dish_count_vars[i] <= 2 * y_vars[i]
    prob += dish_count_vars[i] >= y_vars[i]


# In[10]:


#Constraint Non Jours Successifs : No dish can be included on the menu two days in a row
for product in df.index:
    for jour in JOURS:
        index_jour = JOURS.index(jour)
        if (product, index_jour) in carte_vars:
            prob += p.lpSum(carte_vars[(product, k)] for k in range(max(0, index_jour - 1), min(index_jour+1,6))) <= 1


# In[11]:


# Contraintes quotidiennes par corner et par type d'EPD (de l'onglet Contraintes Jours)

data_d = sheet_contraintes_jours.get('B5:W')
headers_d = ["Corners"]+[f"{header1}_{header2}" for header1 in JOURS for header2 in EPD] #Défintion de header "à la main"
df_d = pd.DataFrame(data_d, columns=headers_d)
nb_corners = df_d.shape[0]

for index_corner in range(nb_corners):
    corner = df_d.loc[index_corner,"Corners"]
    for jour in JOURS:
        for type_EPD in EPD:
            # Permet de mettre la valeur à 0 si la case n'est pas remplie ou l'entier si la case est remplie
            prod_max = 0 if (df_d.loc[index_corner,f'{jour}_{type_EPD}'] == "" or df_d.loc[index_corner,f'{jour}_{type_EPD}'] is None) else int(df_d.loc[index_corner,f'{jour}_{type_EPD}'])
            prob += p.lpSum(carte_vars[(produit, JOURS.index(jour))] for produit in df.index  if df.loc[produit, "EPD"] == type_EPD and df.loc[produit, "Fournisseur"] == corner ) <= prod_max


# In[12]:


# Contraites hebdomadaires par sous categorie (de l'onglet Contraintes Sous Categories)

data_a = sheet_contraintes_sous_categories.get('C4:F')
headers_a = data_a.pop(0)
df_a = pd.DataFrame(data_a, columns=headers_a)
nb_contraintes = df_a.shape[0]


for index_contrainte in range(nb_contraintes) :
    prob += p.lpSum(carte_vars[(produit, index_jour)] for index_jour in range(len(JOURS)) for produit in df.index if df.loc[produit, "EPD"] == df_a.loc[index_contrainte,"Catégorie"] and df.loc[produit, "Fournisseur"] == df_a.loc[index_contrainte,"Fournisseur"] and df.loc[produit, "Sous Categorie Corner"] == df_a.loc[index_contrainte,"Sous Catégorie"] ) <= int(df_a.loc[index_contrainte,"Qté max / semaine"])


# In[13]:


# Contraites quotidiennes par sous categorie (de l'onglet Contraintes Sous Categories)

data_b = sheet_contraintes_sous_categories.get('I4:R')
headers_b = data_b.pop(0)
print(headers_b)
df_b = pd.DataFrame(data_b, columns=headers_b)
nb_contraintes = df_b.shape[0]

for index_contrainte in range(nb_contraintes) :
    for jour in JOURS:
        quantite_max_jour = 0 if (df_b.loc[index_contrainte,jour] == "" or df_b.loc[index_contrainte,jour] is None) else int(df_b.loc[index_contrainte,jour])
        prob += p.lpSum(carte_vars[(produit, JOURS.index(jour))] for produit in df.index if df.loc[produit, "EPD"] == df_b.loc[index_contrainte,"Catégorie"] and df.loc[produit, "Fournisseur"] == df_b.loc[index_contrainte,"Fournisseur"] and df.loc[produit, "Sous Categorie Corner"] == df_b.loc[index_contrainte,"Sous Catégorie"] ) <= quantite_max_jour
    


# In[14]:


# Contraintes particulières hebdomadaires sans conditions (de l'onglet Contraintes Particulières)

data_e = sheet_contraintes_particulieres.get('A2:I')
headers_e = data_e.pop(0)
df_e = pd.DataFrame(data_e, columns=headers_e)
nb_contraintes = df_e.shape[0]

for index_contrainte in range(nb_contraintes):
    if df_e.loc[index_contrainte,"Condition activée ?"] == "TRUE":
        prob += dish_count_vars[int(df_e.loc[index_contrainte,"ID Solver"])]  <= int(df_e.loc[index_contrainte,"Qté max / semaine"])


# In[15]:


# Contraintes particulières hebdomadairess avec conditions (de l'onglet Contraintes Particulières)

data_f = sheet_contraintes_particulieres.get('K2:U')
headers_f = data_f.pop(0)
df_f = pd.DataFrame(data_f, columns=headers_f)
nb_contraintes = df_f.shape[0]

for index_contrainte in range(nb_contraintes):
    if df_f.loc[index_contrainte,"Condition activée ?"] == "TRUE":
        if df_f.loc[index_contrainte,"Fournisseur"] == "Frichti": 
            prob += dish_count_vars[int(df_f.loc[index_contrainte,"ID Solver"])] <= 0
        else : 
            prob += dish_count_vars[int(df_f.loc[index_contrainte,"ID Solver"])] <= 1 - dish_count_vars[int(df_f.loc[index_contrainte,"ID Solver Produit non cumulable"])]
        


# In[16]:


# Contraintes particulières quotidiennes avec conditions (de l'onglet Contraintes Particulières)

data_g = sheet_contraintes_particulieres.get('X2:AG')
headers_g = data_g.pop(0)
df_g = pd.DataFrame(data_g, columns=headers_g)
nb_contraintes = df_g.shape[0]

for index_contrainte in range(nb_contraintes):
    if df_g.loc[index_contrainte,"Condition activée ?"] == "TRUE":
        if df_g.loc[index_contrainte,"Fournisseur Produit non cumulable"] == "Frichti":
            prob += dish_count_vars[int(df_g.loc[index_contrainte,"ID Solver"])] <= 0
        else : 
            prob += dish_count_vars[int(df_g.loc[index_contrainte,"ID Solver"])] <= 1 - dish_count_vars[int(df_g.loc[index_contrainte,"ID Solver Produit non cumulable"])]
        


# In[17]:


# Contrainte d'obligation de mettre à la carte (de l'onglet Contraintes Particulières)

data_h = sheet_contraintes_particulieres.get('AJ2:AQ')
headers_h = data_h.pop(0)
df_h = pd.DataFrame(data_h, columns=headers_h)
nb_contraintes = df_h.shape[0]

for index_contrainte in range(nb_contraintes):
    if df_h.loc[index_contrainte,"Condition activée ?"] == "TRUE":
        prob += dish_count_vars[int(df_h.loc[index_contrainte,"ID Solver"])] == int(df_h.loc[index_contrainte,"Qté semaine"])


# In[18]:


# Solve the problem
prob.solve(solver)

# Check the status of the problem
status = p.LpStatus[prob.status]
print(status)


# In[19]:


# Print the optimal total food cost
print(f"Objective function: {p.value(prob.objective):,.2f}")

# Print the optimal menu for each day
result_list = [] 
for index_jour in range(len(JOURS)):
    for product in df.index:
        if carte_vars[(product, index_jour)].varValue == 1:
            result_dict = {"Day of the week " : index_jour+1,
                           "ID" : df.loc[product, 'ID'],
                          }
            result_list.append(result_dict)

df_result = pd.concat([pd.DataFrame(dict_ID, index=[0]) for dict_ID in result_list], ignore_index=True)

# Print the value of each expression
print("Total food cost: ", p.value(food_cost_expr))

for product in df.index:
    for index_jour in range(len(JOURS)):
        print(f"Valeur de carte_vars[{product}, {index_jour}]: {carte_vars[product, index_jour].varValue}")


# In[20]:


# Write the result back to Google sheets starting from the second row
sheet_output = doc_outil_carte.worksheet("Output")
#Delete the old output
doc_outil_carte.values_clear("Output!C1:D500")

#Paste the new output
sheet_output.update('C1',[df_result.columns.values.tolist()] + df_result.values.tolist())
sheet_output.update('A2', f'{p.value(food_cost_expr):,.2f}')


# In[21]:


for dish, y in y_vars.items():
    print(f"y_{dish} is {y.value()}.")


# In[ ]:





# In[ ]:




