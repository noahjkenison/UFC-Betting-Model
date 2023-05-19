import pandas as pd
import numpy as np
from datetime import *
from dateutil.relativedelta import *

#To add:
#Metrics:
#1. days since last fight?
#2. Different models for different weight classes? 
#3. create model for fighters with minimum 3 ufc fights.


#To create a  model:
#1. "Update database" via scraping.
#2. Generate Advanced stats DF. (using fight database)
#3. Generate Model-ready DB (using advanced stats df)
#4. Use model.generateModel() to generate a model with this dataset.




#To make predictions.
#4. Generate createPrePredictiondf (df of 1 fighter vs another), (From Model-ready DB)
#5. Feed this DF into model using pred_proba() or predict()

def splitCol(column, delimiter):
    x = column.replace("", np.NaN)
    x = column.replace("--", np.NaN)
    dataframe = x.str.split(delimiter, expand=True)
    dataframe = dataframe.astype(float)
    return dataframe

def createAdvStatsdf(fightDB_df, fighterDB_df):
    #Takes up-to-date fight database and fighter database and calculates advanced stats.

    datafolder = 'C:/Users/nkpna/OneDrive/Desktop/Coding/VS Code Projects/UFC project/data/'
    calctemplate = 'prefullStatsCalcTemplate.xlsx'
    postcalc_df = pd.DataFrame()
    calctemplate_df = pd.read_excel(datafolder + "templates/" + calctemplate)

    postcalc_df = calctemplate_df

    fightDB_df['Date'] = pd.to_datetime(fightDB_df['Date'])
    fightDB_df  = fightDB_df.sort_values('Date')


    for col in fightDB_df:
        postcalc_df[col] = fightDB_df[col]

    postcalc_df['FightCount'] = 1

    postcalc_df['Details'] = fightDB_df['Details']

    endTimeCols = splitCol(postcalc_df['Ending Time'], ":")
    ctrlCols = splitCol(postcalc_df['CTRL'], ":")
    #oppctrlCols = splitCol(postcalc_df['OPP CTRL'], ":")

    postcalc_df['CTRL(min)'] = ctrlCols.iloc[:,0]
    postcalc_df['CTRL(sec)'] = ctrlCols.iloc[:,1]
    postcalc_df['CTRL(total min)'] = postcalc_df['CTRL(min)'] + (postcalc_df['CTRL(sec)'] / 60)

    postcalc_df['OPP CTRL(min)'] = ctrlCols.iloc[:,0]
    postcalc_df['OPP CTRL(sec)'] = ctrlCols.iloc[:,1]
    postcalc_df['OPP CTRL(total min)'] = postcalc_df['CTRL(min)'] + (postcalc_df['CTRL(sec)'] / 60)

    postcalc_df['Ending Time(min)'] = endTimeCols.iloc[:,0]
    postcalc_df['Ending Time(sec)'] = endTimeCols.iloc[:,1]

    postcalc_df['Win'] = 0
    postcalc_df['Loss'] = 0

    postcalc_df.loc[postcalc_df['Result'] == 'W', 'Win'] = 1
    postcalc_df.loc[postcalc_df['Result'] == 'L', 'Win'] = 0
    postcalc_df.loc[postcalc_df['Result'] == 'D', 'Win'] = 0
    postcalc_df.loc[postcalc_df['Result'] == 'NC', 'Win'] = 0

    postcalc_df.loc[postcalc_df['Result'] == 'W', 'Loss'] = 0
    postcalc_df.loc[postcalc_df['Result'] == 'L', 'Loss'] = 1
    postcalc_df.loc[postcalc_df['Result'] == 'D', 'Loss'] = 0
    postcalc_df.loc[postcalc_df['Result'] == 'NC', 'Loss'] = 0

    postcalc_df.loc[postcalc_df['Result'] == 'W', 'OPP Win'] = 0
    postcalc_df.loc[postcalc_df['Result'] == 'L', 'OPP Win'] = 1
    postcalc_df.loc[postcalc_df['Result'] == 'D', 'OPP Win'] = 0
    postcalc_df.loc[postcalc_df['Result'] == 'NC', 'OPP Win'] = 0


    postcalc_df.loc[postcalc_df['Result'] == 'W', 'OPP Loss'] = 1
    postcalc_df.loc[postcalc_df['Result'] == 'L', 'OPP Loss'] = 0
    postcalc_df.loc[postcalc_df['Result'] == 'D', 'OPP Loss'] = 0
    postcalc_df.loc[postcalc_df['Result'] == 'NC', 'OPP Loss'] = 0


    postcalc_df.loc[(postcalc_df['Loss'] == 1) & ((postcalc_df['Result Details'] == 'KO/TKO') | (postcalc_df['Result Details'] == "TKO - Doctor's Stoppage")), 'KO REC'] = 1
    postcalc_df.loc[(postcalc_df['Loss'] == 1) & (postcalc_df['Result Details'] == 'Submission') , 'SUB REC'] = 1
    postcalc_df.loc[(postcalc_df['Win'] == 1) & ((postcalc_df['Result Details'] == 'KO/TKO') | (postcalc_df['Result Details'] == "TKO - Doctor's Stoppage")), 'KO GIVEN'] = 1
    postcalc_df.loc[(postcalc_df['Win'] == 1) & (postcalc_df['Result Details'] == 'Submission') , 'SUB GIVEN'] = 1

    postcalc_df.loc[(postcalc_df['OPP Loss'] == 1) & ((postcalc_df['Result Details'] == 'KO/TKO') | (postcalc_df['Result Details'] == "TKO - Doctor's Stoppage")), 'OPP KO REC'] = 1
    postcalc_df.loc[(postcalc_df['OPP Loss'] == 1) & (postcalc_df['Result Details'] == 'Submission') , 'OPP SUB REC'] = 1
    postcalc_df.loc[(postcalc_df['OPP Win'] == 1) & ((postcalc_df['Result Details'] == 'KO/TKO') | (postcalc_df['Result Details'] == "TKO - Doctor's Stoppage")), 'OPP KO GIVEN'] = 1
    postcalc_df.loc[(postcalc_df['OPP Win'] == 1) & (postcalc_df['Result Details'] == 'Submission') , 'OPP SUB GIVEN'] = 1



    postcalc_df['KO REC'] = postcalc_df['KO REC'].replace(np.NaN, 0)
    postcalc_df['SUB REC'] = postcalc_df['SUB REC'].replace(np.NaN, 0)
    postcalc_df['SUB GIVEN'] = postcalc_df['SUB GIVEN'].replace(np.NaN, 0)
    postcalc_df['KO GIVEN'] = postcalc_df['KO GIVEN'].replace(np.NaN, 0)

    postcalc_df['OPP KO REC'] = postcalc_df['OPP KO REC'].replace(np.NaN, 0)
    postcalc_df['OPP SUB REC'] = postcalc_df['OPP SUB REC'].replace(np.NaN, 0)
    postcalc_df['OPP SUB GIVEN'] = postcalc_df['OPP SUB GIVEN'].replace(np.NaN, 0)
    postcalc_df['OPP KO GIVEN'] = postcalc_df['OPP KO GIVEN'].replace(np.NaN, 0)

    postcalc_df['SIG STR DIF'] = postcalc_df['SIG STR Landed'] - postcalc_df['OPP SIG STR Landed']
    postcalc_df['OPP SIG STR DIF'] = postcalc_df['OPP SIG STR Landed'] - postcalc_df['SIG STR Landed'] 

    postcalc_df['TD LANDED DIF'] = postcalc_df['TD LANDED'] - postcalc_df['OPP TD LANDED']
    postcalc_df['OPP TD LANDED DIF'] = postcalc_df['OPP TD LANDED'] - postcalc_df['TD LANDED']

    postcalc_df['CTRL DIF(min)'] = postcalc_df['CTRL(total min)'] - postcalc_df['OPP CTRL(total min)']
    postcalc_df['OPP CTRL DIF(min)'] = postcalc_df['OPP CTRL(total min)'] - postcalc_df['CTRL(total min)']


    postcalc_df['Title Fight'] = 0

    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Flyweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Bantamweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Featherweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Lightweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Welterweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Middleweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Light Heavyweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Heavyweight Title Bout") , 'Title Fight'] = 1

    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Flyweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Bantamweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Featherweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Lightweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Welterweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Middleweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Light Heavyweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Heavyweight Title Bout") , 'Title Fight'] = 1

    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Women's Strawweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Women's Flyweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Bantamweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Featherweight Title Bout") , 'Title Fight'] = 1

    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Women's Strawweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Women's Flyweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Bantamweight Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC Interim Featherweight Title Bout") , 'Title Fight'] = 1

    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 2 Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 3 Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 4 Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 5 Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 6 Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 7 Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 9 Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 10 Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 13 Heavyweight Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 13 Lightweight Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 14 Middleweight Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 15 Heavyweight Tournament Title Bout") , 'Title Fight'] = 1
    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 17 Middleweight Tournament Title Bout") , 'Title Fight'] = 1

    postcalc_df['Title Fight Win'] = 0
    postcalc_df['OPP TITLE FIGHT WIN'] = 0

    postcalc_df.loc[(postcalc_df['Title Fight'] == 1) & (postcalc_df['Win'] == 1) , 'Title Fight Win'] = 1
    postcalc_df.loc[(postcalc_df['Title Fight'] == 1) & (postcalc_df['OPP Win'] == 1) , 'OPP TITLE FIGHT WIN'] = 1


    postcalc_df.loc[(postcalc_df['Time Format'] == "2 Rnd (5-5)") , 'ROUNDS:'] = 2
    postcalc_df.loc[(postcalc_df['Time Format'] == "3 Rnd (5-5-5)") , 'ROUNDS:'] = 3
    postcalc_df.loc[(postcalc_df['Time Format'] == "3 Rnd + OT (5-5-5-5)") , 'ROUNDS:'] = 4
    postcalc_df.loc[(postcalc_df['Time Format'] == "5 Rnd (5-5-5-5-5)") , 'ROUNDS:'] = 5

    postcalc_df.loc[(postcalc_df['Weight Class'] == "UFC 2 Tournament Title Bout") , 'Title Fight'] = 1

    merge1df =   pd.merge(postcalc_df, fighterDB_df, suffixes=('','-dup'), on ='Fighter', how ='left')


    merge1df = merge1df.drop(columns = ['Fighter.1', 'SIG STR Landed.1', 'SIG STR ATT.1', 'SIG STR %.1'])
    merge1df = merge1df.drop(columns = ['OPP NAME.1', 'OPP SIG STR Landed.1', 'OPP SIG STR ATT.1', 'OPP SIG STR %.1', 'Link'])                      

    merge2df = pd.merge(merge1df, fighterDB_df, suffixes=('','-OPP'), left_on="OPP NAME", right_on = "Fighter", how ='left')

    merge2df = merge2df.drop(columns = ['Link', 'Fighter-OPP' ])            
    merge2df.rename(columns = {'DOB-OPP':'OPP DOB', 'HT-OPP':'OPP HT', 'REACH-OPP': 'OPP REACH', 'STANCE-OPP': 'OPP STANCE', 'WEIGHT-OPP': 'OPP WEIGHT'}, inplace = True)


    merge2df['HT'] = merge2df['HT'].str.replace("'", ":")
    merge2df['HT'] = merge2df['HT'].str.replace('"', '')
    HTcols = splitCol(merge2df['HT'], ":")

    merge2df['HT(FT)'] = HTcols.iloc[:,0]
    merge2df['HT(IN)'] = HTcols.iloc[:,1]
    merge2df['HT(TOTAL)'] = (merge2df['HT(FT)'] * 12) + merge2df['HT(IN)']

    merge2df['OPP HT'] = merge2df['OPP HT'].str.replace("'", ":")
    merge2df['OPP HT'] = merge2df['OPP HT'].str.replace('"', '')
    OPPHTcols = splitCol(merge2df['OPP HT'], ":")

    merge2df['OPP HT(FT)'] = OPPHTcols.iloc[:,0]
    merge2df['OPP HT(IN)'] = OPPHTcols.iloc[:,1]
    merge2df['OPP HT(TOTAL)'] = (merge2df['OPP HT(FT)'] * 12) + merge2df['OPP HT(IN)']

    merge2df['HEIGHT DIFFERENTIAL'] = merge2df['HT(TOTAL)'] - merge2df['OPP HT(TOTAL)']


    merge2df['REACH'] = merge2df['REACH'].str.replace('"', '')
    merge2df['OPP REACH'] = merge2df['OPP REACH'].str.replace('"', '')

    merge2df['REACH'] = merge2df['REACH'].replace("--", np.NaN)
    merge2df['OPP REACH']  = merge2df['OPP REACH'].replace("--", np.NaN)

    merge2df['REACH'] = merge2df['REACH'].astype(float)
    merge2df['OPP REACH'] = merge2df['OPP REACH'].astype(float)


    merge2df['REACH DIFFERENTIAL'] = merge2df['REACH'] - merge2df['OPP REACH']

    merge2df['FIGHT DURATION (MIN)'] = np.where(merge2df['Ending Round']== 1, merge2df['Ending Time(min)'] + (merge2df['Ending Time(sec)'] / 60), 
                                                ((merge2df['Ending Round'] - 1) * 5) + merge2df['Ending Time(min)'] + (merge2df['Ending Time(sec)'] / 60))


    merge2df['DOB'] = merge2df['DOB'].replace("--", np.NaN)
    merge2df['OPP DOB'] = merge2df['OPP DOB'].replace("--", np.NaN)


    merge2df['DOB'] = pd.to_datetime(merge2df['DOB'])
    merge2df['OPP DOB'] = pd.to_datetime(merge2df['OPP DOB'])

    merge2df['FIGHTER AGE'] = ((merge2df['Date'] - merge2df['DOB']).dt.days) / 365
    merge2df['OPP AGE'] = ((merge2df['Date'] - merge2df['OPP DOB']).dt.days) / 365
    merge2df['AGE DIFFERENTIAL'] = merge2df['FIGHTER AGE'] - merge2df['OPP AGE']

    merge2df.loc[merge2df['Weight Class'].str.contains('Flyweight'), "Weight Class Combined"] = "Flyweight"
    merge2df.loc[merge2df['Weight Class'].str.contains('Bantamweight'), "Weight Class Combined"] = "Bantamweight"
    merge2df.loc[merge2df['Weight Class'].str.contains('Featherweight'), "Weight Class Combined"] = "Featherweight"
    merge2df.loc[merge2df['Weight Class'].str.contains('Lightweight'), "Weight Class Combined"] = "Lightweight"
    merge2df.loc[merge2df['Weight Class'].str.contains('Welterweight'), "Weight Class Combined"] = "Welterweight"
    merge2df.loc[merge2df['Weight Class'].str.contains('Middleweight'), "Weight Class Combined"] = "Middleweight"
    merge2df.loc[merge2df['Weight Class'].str.contains('Light Heavyweight'), "Weight Class Combined"] = "Light Heavyweight"
    merge2df.loc[merge2df['Weight Class'].str.contains('Heavyweight'), "Weight Class Combined"] = "Heavyweight"
    merge2df.loc[merge2df['Weight Class'].str.contains('Super Heavyweight'), "Weight Class Combined"] = "Super Heavyweight"
    merge2df.loc[merge2df['Weight Class'].str.contains('Openweight'), "Weight Class Combined"] = "Openweight"
    merge2df.loc[merge2df['Weight Class'].str.contains('Catch Weight'), "Weight Class Combined"] = "Catch Weight"

    merge2df.loc[merge2df['Weight Class'].str.contains("Women's Strawweight"), "Weight Class Combined"] = "Women's Strawweight"
    merge2df.loc[merge2df['Weight Class'].str.contains("Women's Flyweight"), "Weight Class Combined"] = "Women's Flyweight"
    merge2df.loc[merge2df['Weight Class'].str.contains("Women's Bantamweight"), "Weight Class Combined"] = "Women's Bantamweight"
    merge2df.loc[merge2df['Weight Class'].str.contains("Women's Featherweight"), "Weight Class Combined"] = "Women's Featherweight"

    #Troubleshooting notes.
    # x  = merge2df['Win']

    # print('this is my type, before changing anything')
    # print(type(x))
    # print(type(x.iloc[2]))

    # x  = merge2df['Win']

    # print('this is my type, setting win straight to 1')
    # print(type(x))
    # print(type(x.iloc[2]))


    # x = merge2df['Win']
    # print('this is my type, setting win column with conditional.')
    # print(type(x))
    # print(type(x.iloc[2]))

    
    fightergroups = merge2df.groupby(['Fighter'])
    opponentgroups = merge2df.groupby(['OPP NAME'])

    merge2df['SIG STR Landed / MIN'] = merge2df['SIG STR Landed'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['SIG STR ATT / MIN'] = merge2df['SIG STR ATT'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['TOTAL STR ATT / MIN'] = merge2df['TOTAL STR ATT'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['TOTAL STR LANDED / MIN'] = merge2df['TOTAL STR LANDED'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['TD LANDED / MIN'] = merge2df['TD LANDED'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['TD ATT / MIN'] = merge2df['TD ATT'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['SUB ATT / MIN'] = merge2df['SUB ATT'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['SIG STR DIF / MIN'] = merge2df['SIG STR DIF'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['TD LANDED DIF / MIN'] = merge2df['TD LANDED'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['SIG STR REC / MIN'] = merge2df['OPP SIG STR Landed'] / merge2df['FIGHT DURATION (MIN)']

    merge2df['OPP SIG STR Landed / MIN'] = merge2df['OPP SIG STR Landed'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['OPP SIG STR ATT / MIN'] = merge2df['OPP SIG STR ATT'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['OPP TOTAL STR ATT / MIN'] = merge2df['OPP TOTAL STR ATT'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['OPP TOTAL STR LANDED / MIN'] = merge2df['OPP TOTAL STR LANDED'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['OPP TD LANDED / MIN'] = merge2df['OPP TD LANDED'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['OPP TD ATT / MIN'] = merge2df['OPP TD ATT'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['OPP SUB ATT / MIN'] = merge2df['OPP SUB ATT'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['OPP SIG STR DIF / MIN'] = merge2df['OPP SIG STR DIF'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['OPP TD LANDED DIF / MIN'] = merge2df['OPP TD LANDED'] / merge2df['FIGHT DURATION (MIN)']
    merge2df['OPP SIG STR REC / MIN'] = merge2df['SIG STR Landed'] / merge2df['FIGHT DURATION (MIN)']

    merge2df['(PF) CAREER WINS'] = (fightergroups['Win'].cumsum())
    merge2df['(PF) CAREER LOSSES'] = (fightergroups['Loss'].cumsum())
    merge2df['(PF) CAREER UFC FIGHTS'] = (fightergroups['FightCount'].cumsum()) # could do the same thing... make a column of "1s".
    merge2df['(PF) CAREER KO REC'] = (fightergroups['KO REC'].cumsum())
    merge2df['(PF) CAREER SUB REC'] = (fightergroups['SUB REC'].cumsum())
    merge2df['(PF) CAREER KO GIVEN'] = (fightergroups['KO GIVEN'].cumsum())
    merge2df['(PF) CAREER SUB GIVEN'] = (fightergroups['SUB GIVEN'].cumsum())
    merge2df['(PF) CAREER WIN %'] = merge2df['(PF) CAREER WINS'] / merge2df['(PF) CAREER UFC FIGHTS']
    merge2df['(PF) CAREER KD REC'] = (fightergroups['OPP KD'].cumsum())
    merge2df['(PF) CAREER FINISHES'] = merge2df['(PF) CAREER KO GIVEN'] + merge2df['(PF) CAREER SUB GIVEN']

    merge2df['(PF) OPP CAREER WINS'] = (opponentgroups['Loss'].cumsum())
    merge2df['(PF) OPP CAREER LOSSES'] = (opponentgroups['Win'].cumsum())
    merge2df['(PF) OPP CAREER UFC FIGHTS'] = (opponentgroups['FightCount'].cumsum()) # could do the same thing... make a column of "1s".
    merge2df['(PF) OPP CAREER KO REC'] = (opponentgroups['OPP KO REC'].cumsum() )
    merge2df['(PF) OPP CAREER SUB REC'] = (opponentgroups['OPP SUB REC'].cumsum())
    merge2df['(PF) OPP CAREER KO GIVEN'] = (opponentgroups['OPP KO GIVEN'].cumsum())
    merge2df['(PF) OPP CAREER SUB GIVEN'] = (opponentgroups['OPP SUB GIVEN'].cumsum())
    merge2df['(PF) OPP CAREER WIN %'] = merge2df['(PF) OPP CAREER WINS'] / merge2df['(PF) OPP CAREER UFC FIGHTS']
    merge2df['(PF) OPP CAREER KD REC'] = (opponentgroups['KD'].cumsum())
    merge2df['(PF) OPP CAREER FINISHES'] = merge2df['(PF) OPP CAREER KO GIVEN'] + merge2df['(PF) OPP CAREER SUB GIVEN']


    merge2df['(PF) WIN BY KO %'] =  merge2df['(PF) CAREER KO GIVEN'] / merge2df['(PF) CAREER UFC FIGHTS']
    merge2df['(PF) WIN BY SUB %'] =  merge2df['(PF) CAREER SUB GIVEN'] / merge2df['(PF) CAREER UFC FIGHTS']
    merge2df['(PF) LOSS BY KO %'] =  merge2df['(PF) CAREER KO REC'] / merge2df['(PF) CAREER UFC FIGHTS']
    merge2df['(PF) LOSS BY SUB %'] =  merge2df['(PF) CAREER SUB REC'] / merge2df['(PF) CAREER UFC FIGHTS']

    merge2df['(PF) OPP WIN BY KO %'] =  merge2df['(PF) OPP CAREER KO GIVEN'] / merge2df['(PF) OPP CAREER UFC FIGHTS']
    merge2df['(PF) OPP WIN BY SUB %'] =  merge2df['(PF) OPP CAREER SUB GIVEN'] / merge2df['(PF) OPP CAREER UFC FIGHTS']
    merge2df['(PF) OPP LOSS BY KO %'] =  merge2df['(PF) OPP CAREER KO REC'] / merge2df['(PF) OPP CAREER UFC FIGHTS']
    merge2df['(PF) OPP LOSS BY SUB %'] =  merge2df['(PF) OPP CAREER SUB REC'] / merge2df['(PF) OPP CAREER UFC FIGHTS']


    merge2df['(PF) OPP # WINS IN LAST 3'] = opponentgroups['Loss'].rolling(3, closed='left').sum().reset_index(0, drop = True)
    merge2df['(PF) OPP # LOSS IN LAST 3'] = opponentgroups['Win'].rolling(3, closed='left').sum().reset_index(0, drop = True) 
    merge2df['(PF) OPP WIN % LAST 3'] = merge2df['(PF) OPP # WINS IN LAST 3'] / 3
    merge2df['(PF) OPP # FINISHES IN LAST 3'] = opponentgroups['OPP KO GIVEN'].rolling(3, closed='left').sum().reset_index(0, drop = True) + opponentgroups['OPP SUB GIVEN'].rolling(3, closed='left').sum().reset_index(0, drop = True) 

    merge2df['(PF) # WINS IN LAST 3'] = fightergroups['Win'].rolling(3, closed='left').sum().reset_index(0, drop = True)
    merge2df['(PF) # LOSS IN LAST 3'] = fightergroups['Loss'].rolling(3, closed='left').sum().reset_index(0, drop = True) 
    merge2df['(PF) WIN % LAST 3'] = merge2df['(PF) # WINS IN LAST 3'] / 3
    merge2df['(PF) # FINISHES IN LAST 3'] = fightergroups['KO GIVEN'].rolling(3, closed='left').sum().reset_index(0, drop = True) + fightergroups['SUB GIVEN'].rolling(3, closed='left').sum().reset_index(0, drop = True) 

    merge2df['(PF) # WINS IN LAST 5'] = fightergroups['Win'].rolling(5).sum().reset_index(0, drop = True)
    merge2df['(PF) # LOSS IN LAST 5'] = fightergroups['Loss'].rolling(5).sum().reset_index(0, drop = True)
    merge2df['(PF) WIN % LAST 5'] = merge2df['(PF) # WINS IN LAST 5'] / 5
    merge2df['(PF) # FINISHES IN LAST 5'] = fightergroups['KO GIVEN'].rolling(5, closed='left').sum().reset_index(0, drop = True) + fightergroups['SUB GIVEN'].rolling(5, closed='left').sum().reset_index(0, drop = True) 

    merge2df['(PF) OPP # WINS IN LAST 5'] = opponentgroups['Loss'].rolling(5).sum().reset_index(0, drop = True)
    merge2df['(PF) OPP # LOSS IN LAST 5'] = opponentgroups['Win'].rolling(5).sum().reset_index(0, drop = True)
    merge2df['(PF) OPP WIN % LAST 5'] = merge2df['(PF) OPP # WINS IN LAST 5'] / 5
    merge2df['(PF) OPP # FINISHES IN LAST 5'] = opponentgroups['OPP KO GIVEN'].rolling(5, closed='left').sum().reset_index(0, drop = True) + opponentgroups['OPP SUB GIVEN'].rolling(5, closed='left').sum().reset_index(0, drop = True) 




#####STATS USED FOR PREDICTION#############

    merge2df['(PF) SIG STR DIF / MIN MA3'] = fightergroups['SIG STR DIF / MIN'].rolling(window=3,center=False, min_periods = 3).mean().reset_index(0, drop = True)
    merge2df['(PF) TD DIF / MIN MA3'] = fightergroups['TD LANDED DIF / MIN'].rolling(window=3,center=False, min_periods = 3).mean().reset_index(0, drop = True)
    merge2df['(PF) CTRL DIF / MIN MA3'] = fightergroups['CTRL DIF(min)'].rolling(window=3,center=False, min_periods = 3).mean().reset_index(0, drop = True)
    merge2df['(PF) TOTAL STR ATT / MIN MA3'] = fightergroups['TOTAL STR ATT / MIN'].rolling(window=3,center=False, min_periods = 3).mean().reset_index(0, drop = True)

    merge2df['(PF) SIG STR DIF / MIN MA5'] = fightergroups['SIG STR DIF / MIN'].rolling(window=5,center=False, min_periods = 5).mean().reset_index(0, drop = True)
    merge2df['(PF) TD DIF / MIN MA5'] = fightergroups['TD LANDED DIF / MIN'].rolling(window=5,center=False, min_periods = 5).mean().reset_index(0, drop = True)
    merge2df['(PF) CTRL DIF / MIN MA5'] = fightergroups['CTRL DIF(min)'].rolling(window=5,center=False, min_periods = 5).mean().reset_index(0, drop = True)
    merge2df['(PF) TOTAL STR ATT / MIN MA5'] = fightergroups['TOTAL STR ATT / MIN'].rolling(window=5,center=False, min_periods = 5).mean().reset_index(0, drop = True)


    merge2df['(PF) OPP SIG STR DIF / MIN MA3'] = opponentgroups['OPP SIG STR DIF / MIN'].rolling(window=3,center=False, min_periods = 3).mean().reset_index(0, drop = True)
    merge2df['(PF) OPP TD DIF / MIN MA3'] = opponentgroups['OPP TD LANDED DIF / MIN'].rolling(window=3,center=False, min_periods = 3 ).mean().reset_index(0, drop = True)
    merge2df['(PF) OPP CTRL DIF / MIN MA3'] = opponentgroups['OPP CTRL DIF(min)'].rolling(window=3,center=False, min_periods = 3).mean().reset_index(0, drop = True)
    merge2df['(PF) OPP TOTAL STR ATT / MIN MA3'] = opponentgroups['OPP TOTAL STR ATT / MIN'].rolling(window=3,center=False, min_periods = 3).mean().reset_index(0, drop = True)

    merge2df['(PF) OPP SIG STR DIF / MIN MA5'] = opponentgroups['OPP SIG STR DIF / MIN'].rolling(window=5,center=False, min_periods = 5).mean().reset_index(0, drop = True)
    merge2df['(PF) OPP TD DIF / MIN MA5'] = opponentgroups['OPP TD LANDED DIF / MIN'].rolling(window=5,center=False, min_periods = 5).mean().reset_index(0, drop = True)
    merge2df['(PF) OPP CTRL DIF / MIN MA5'] = opponentgroups['OPP CTRL DIF(min)'].rolling(window=5,center=False, min_periods = 5).mean().reset_index(0, drop = True)
    merge2df['(PF) OPP TOTAL STR ATT / MIN MA5'] = opponentgroups['OPP TOTAL STR ATT / MIN'].rolling(window=5,center=False, min_periods = 5).mean().reset_index(0, drop = True)




    #Different way to calculate these values as well...
    #EX: newdf['CAREER SIG STR ABS / MIN'] = fightergroups['OPP SIG STR Landed'].cumsum() / fightergroups['FIGHT DURATION (MIN)'].cumsum()
    #EX: newdf['CAREER SIG STR ABS / MIN2'] =  fightergroups['SIG STR DIF / MIN'].cumsum() / newdf['(PF) CAREER UFC FIGHTS']


    merge2df['(PF) CAREER SIG STR DIF / MIN'] =  fightergroups['SIG STR DIF / MIN'].cumsum() / merge2df['(PF) CAREER UFC FIGHTS']
    merge2df['(PF) CAREER TD LANDED / MIN'] =  fightergroups['TD LANDED DIF / MIN'].cumsum() / merge2df['(PF) CAREER UFC FIGHTS']
    merge2df['(PF) CAREER SIG STR ABS'] =  fightergroups['OPP SIG STR Landed'].cumsum() 
    merge2df['(PF) CAREER CTRL DIF'] =  fightergroups['CTRL DIF(min)'].cumsum() / fightergroups['FIGHT DURATION (MIN)'].cumsum()
    merge2df['(PF) CAREER SIG STR ABS / MIN'] =  fightergroups['SIG STR REC / MIN'].cumsum() / merge2df['(PF) CAREER UFC FIGHTS']
    merge2df['(PF) CAREER TOTAL STR ATT / MIN'] = fightergroups['TOTAL STR ATT / MIN'].cumsum() / merge2df['(PF) CAREER UFC FIGHTS']
    merge2df['(PF) CAREER TITLE FIGHT WINS'] = fightergroups['Title Fight Win'].cumsum()



    merge2df['(PF) OPP CAREER SIG STR DIF / MIN'] =  opponentgroups['OPP SIG STR DIF / MIN'].cumsum() / merge2df['(PF) OPP CAREER UFC FIGHTS']
    merge2df['(PF) OPP CAREER TD LANDED / MIN'] =  opponentgroups['OPP TD LANDED DIF / MIN'].cumsum() / merge2df['(PF) OPP CAREER UFC FIGHTS']
    merge2df['(PF) OPP CAREER SIG STR ABS'] =  opponentgroups['SIG STR Landed'].cumsum() 
    merge2df['(PF) OPP CAREER CTRL DIF'] =  opponentgroups['OPP CTRL DIF(min)'].cumsum() / fightergroups['FIGHT DURATION (MIN)'].cumsum()
    merge2df['(PF) OPP CAREER SIG STR ABS / MIN'] =  opponentgroups['OPP SIG STR REC / MIN'].cumsum() / merge2df['(PF) OPP CAREER UFC FIGHTS']
    merge2df['(PF) OPP CAREER TOTAL STR ATT / MIN'] = opponentgroups['OPP TOTAL STR ATT / MIN'].cumsum() / merge2df['(PF) OPP CAREER UFC FIGHTS']
    merge2df['(PF) OPP CAREER TITLE FIGHT WINS'] = opponentgroups['OPP TITLE FIGHT WIN'].cumsum()

    return merge2df

def createPrePredictiondf(fightDB_df, fighter1, fighter2, fightdate, rounds, weightclass):
    #Creates a DF that includes fighter1 stats vs fighter2 stats. This can be fed directly into a model to find a prediction.
    
    
    print('printing fighters!')
    print(fighter1)
    print(fighter2)

    fighterX_df = pd.read_excel('C:/Users/nkpna/OneDrive/Desktop/Coding/VS Code Projects/UFC project/data/templates/' + 'predTemplate2.xlsx')

    #These DFs split all the "Fighter stats" from the "OPP stats"
    oppdftest = fightDB_df.iloc[:, fightDB_df.columns.str.contains('OPP')]
    fighterdftest = fightDB_df.iloc[:, ~fightDB_df.columns.str.contains('OPP')]

    fighter1df = fighterdftest.loc[(fighterdftest['Fighter'] == fighter1)]
    fighter2df = oppdftest.loc[(oppdftest['OPP NAME'] == fighter2)]

    fighter1df = fighter1df.tail(1)
    fighter2df = fighter2df.tail(1)

    print(fighter1df)
    print(fighter2df)

    # fighter1header = list(fighter1df)
    # fighter2header = list(fighter2df)
    fightDB_dfheader = list(fightDB_df)


    for col in fightDB_dfheader:
        if col in fighterX_df and col in fighter1df:
            fighterX_df[col] = fighter1df[col]
        if col in fighterX_df and col in fighter2df:
            fighterX_df[col] = fighter2df.loc[:, col].values


    fighterDOB = pd.to_datetime(fighterX_df['DOB'].values[0], format ='%b %d, %Y')
    oppDOB = pd.to_datetime(fighterX_df['OPP DOB'].values[0], format ='%b %d, %Y')

    fighterAge = relativedelta(fightdate, fighterDOB).years
    oppAge = relativedelta(fightdate, oppDOB).years
    ageDifferential = fighterAge - oppAge


    fighterX_df['FIGHTER AGE'] = fighterAge
    fighterX_df['AGE DIFFERENTIAL'] = ageDifferential
    fighterX_df['ROUNDS:'] = rounds

    fighterX_df['HEIGHT DIFFERENTIAL'] = fighterX_df['HT(TOTAL)'] - fighterX_df['OPP HT(TOTAL)'] 
    fighterX_df['REACH DIFFERENTIAL'] = fighterX_df['REACH'] - fighterX_df['OPP REACH'] 

    fighterX_df.STANCE.replace('Orthodox', 1, inplace=True)
    fighterX_df.STANCE.replace('Southpaw', 2, inplace=True)
    fighterX_df.STANCE.replace('Switch', 3, inplace=True)
    fighterX_df.STANCE.replace('Open Stance', 4, inplace=True)

    fighterX_df['Weight Class Combined'] = weightclass
    fighterX_df = codeWeightClasses(fighterX_df)

    fighterX_df = fighterX_df.drop(labels=['Result', 'DOB', 'Date', 'Fighter', 
                                           'OPP DOB', 'OPP HT(TOTAL)', 'OPP NAME', 'OPP REACH'], axis=1)



    #fighterX_df is ready to be inputted into our model as the "X" variable
    return fighterX_df


def createModelDBdf(advancedstatsdb_df, dftemplate):
    #Takes Advanced Data df and cleans it / picks which stats we want to include in model. 
    #drops stats from *current* fight. ex: sig strikes from CURRENT fight. (as in, prepares the df to be put into the model)
    
    modelDB_df = dftemplate
    dfheader = list(advancedstatsdb_df)

    #Template used to determine which columns to include in our model.
    for col in dfheader:
        if col in modelDB_df and col in advancedstatsdb_df:
            modelDB_df[col] = advancedstatsdb_df[col]


    modelDB_df.Result.replace('W', 1, inplace=True)
    modelDB_df.Result.replace('L', 0, inplace=True)
    modelDB_df.Result.replace('D', 0, inplace=True)
    modelDB_df.Result.replace('NC', 0, inplace=True)

    modelDB_df.STANCE.replace('Orthodox', 1, inplace=True)
    modelDB_df.STANCE.replace('Southpaw', 2, inplace=True)
    modelDB_df.STANCE.replace('Switch', 3, inplace=True)
    modelDB_df.STANCE.replace('Open Stance', 4, inplace=True)

    modelDB_df = codeWeightClasses(modelDB_df)

    modelDB_df = modelDB_df.replace([np.inf, -np.inf], np.nan)
    modelDB_df = modelDB_df.replace('', np.nan)
    modelDB_df = modelDB_df.dropna()

    #Regturns a DF that is compatible with model.
    return modelDB_df

def dropMA5Test(df):
    df = df.drop(labels=['(PF) # WINS IN LAST 5', '(PF) # LOSS IN LAST 5', '(PF) WIN % LAST 5', '(PF) # FINISHES IN LAST 5', 
                                           '(PF) OPP # WINS IN LAST 5', '(PF) OPP # LOSS IN LAST 5', '(PF) OPP WIN % LAST 5', '(PF) OPP # FINISHES IN LAST 5',
                                           '(PF) SIG STR DIF / MIN MA5', '(PF) TD DIF / MIN MA5', '(PF) CTRL DIF / MIN MA5', '(PF) TOTAL STR ATT / MIN MA5',
                                           '(PF) OPP SIG STR DIF / MIN MA5', '(PF) OPP TD DIF / MIN MA5', '(PF) OPP CTRL DIF / MIN MA5', '(PF) OPP TOTAL STR ATT / MIN MA5'], axis=1)
    
    return df
    
#Converts American odds (+150, -300, etc.) to decimal/percentage odds (.75, .82, etc.)
def AmericanOddstoDecimal(americanodds):
    if (americanodds > 0):
        decimalodds = ((100 / (americanodds + 100)) * 100)

    if (americanodds < 0):
        decimalodds = (((americanodds * -1) / ((americanodds * -1) + 100)) * 100)

    if (americanodds == 0):
        decimalodds = 0
        
    return decimalodds

#Reverse of above.
def decimalOddsToAmerican(decimalodds):
    #converting to a -- american odd (favorite)
    if(decimalodds >= .50):
        #americanodds = (decimalodds / (100 - decimalodds)) * -100
        americanodds = (100 * decimalodds) / ( 1 - decimalodds)
        americanodds = americanodds * -1

    #converting to a ++ american odd (underdog)
    if(decimalodds < .50):
        americanodds = (100 / decimalodds) - 100
        
    americanodds = americanodds.astype(int)

    return americanodds

#Encodes weight classes so they can be used in model.
def codeWeightClasses(df):
    df['Weight Class Combined'].replace('Flyweight', 1, inplace=True)
    df['Weight Class Combined'].replace('Bantamweight', 2, inplace=True)
    df['Weight Class Combined'].replace('Featherweight', 3, inplace=True)
    df['Weight Class Combined'].replace('Lightweight', 4, inplace=True)
    df['Weight Class Combined'].replace('Welterweight', 5, inplace=True)
    df['Weight Class Combined'].replace('Middleweight', 6, inplace=True)
    df['Weight Class Combined'].replace('Light Heavyweight', 7, inplace=True)
    df['Weight Class Combined'].replace('Heavyweight', 8, inplace=True)
    df['Weight Class Combined'].replace("Women's Strawweight", 9, inplace=True)
    df['Weight Class Combined'].replace("Women's Flyweight", 10, inplace=True)
    df['Weight Class Combined'].replace("Women's Bantamweight", 11, inplace=True)
    df['Weight Class Combined'].replace("Women's Featherweight", 12, inplace=True)
    df['Weight Class Combined'].replace("Catch Weight", 13, inplace=True)
    return df

#Takes prediction score and returns a bet recommendation to user.
def GetBetRecommendation(predictionscore):

    if(predictionscore < -5):
        recommendation = "DO NOT BET"

    if(predictionscore < 0 and predictionscore >= -5):
        recommendation = "User decision, take into account non-computable factors"


    if(predictionscore == 0):
        recommendation = "Missing site odds or mdl odds. No rec possible."

    if(predictionscore > 0 and predictionscore <5):
        recommendation = "User decision, slightly good bet."

    if(predictionscore >= 5 and predictionscore <10):
        recommendation = "BET 2U"

    if (predictionscore >= 10 and predictionscore <15):
        recommendation = "BET 4U"

    if (predictionscore >= 15 and predictionscore <25):
        recommendation = "BET 5U"

    if (predictionscore > 25 and predictionscore < 35):
        recommendation = "BET 7U"
    
    if predictionscore >= 35:
        recommendation = "BET 10U or more"
    
    return recommendation

#Generates a prediction score based off the difference between our model's expectation for the fight and the betting website's odds.
def PredictionScore(myodds, siteodds):    

    if(siteodds == 0 or myodds == 0):
        score = 0

    else:
        score = AmericanOddstoDecimal(myodds) - AmericanOddstoDecimal(siteodds)

    return score

def GetPredictionScoreColor(predictionscore):

    if(predictionscore <= -5):
        color = "green"

    if(predictionscore > -5 and predictionscore <= -3):
        color = "green"


    if(predictionscore > -3 and predictionscore <= -2 ):
        color = "green"


    if(predictionscore > -2 and predictionscore < -1 ):
        color = "green"

    if(predictionscore <= -1 or predictionscore <= 1):
        color = "white"

    if(predictionscore > 1):
        color = "red"

    if(predictionscore == 'inf' or predictionscore == '-inf' ):
        color = "white"


    return color

#Calculates the $ return on one unit bet.
def calculateOneUnitReturn(siteodds, unitsize):
    if(siteodds> 0):
        oneunitreturn = (siteodds / 100) * unitsize

    if(siteodds < 0):
        oneunitreturn = unitsize / (siteodds / 100)

    if(siteodds == 0):
        oneunitreturn = 9999

    oneunitreturn = round(oneunitreturn)
    oneunitreturn = abs(oneunitreturn)

    return oneunitreturn

#Calculates the total return on a bet.
def calculateReturn(siteodds, unitsize, unitAmt):
    oneUnit = calculateOneUnitReturn(siteodds, unitsize)

    

    return(abs(round(oneUnit * unitAmt)))


        #The lower the score, the better?
        #maybe also return how much 1U would return on siteodds?


def addPredictiontoDB(new_df, predictionsdb_df):

    #concat then remove duplicates?
    predictionsdb_df = pd.concat([predictionsdb_df, new_df], axis=0)
    predictionsdb_df = predictionsdb_df.drop_duplicates()

    return predictionsdb_df

class Predictions_DB:
    def __init__(self, predTBD_df, predCOMPLETE_df):
        self.predTBD_df = predTBD_df
        self.predCOMPLETE_df = predCOMPLETE_df

    def updatePredictionsResults(self, fightdb_df):

        xdf = fightdb_df.loc[(fightdb_df['Fighter'] == self.predTBD_df['FIGHTER']) & (fightdb_df['OPP NAME'] == self.predTBD_df['OPPONENT']) & (fightdb_df['Date'] == self.predTBD_df['FIGHT DATE'])]
        return xdf


        #df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison')]


def updatePredictionsResults(fightdb_df, predTBD_df, predCOMPLETE_df):
        # xdf = fightdb_df.loc[(fightdb_df['Fighter'] == 'Jan Blachowicz') & (fightdb_df['OPP NAME'] == 'Magomed Ankalaev') & (fightdb_df['Date'] == predTBD_df['December 10, 2022'])]
        xdf = fightdb_df.loc[(fightdb_df['Fighter'] == 'Jan Blachowicz') & fightdb_df['OPP NAME'] == 'Magomed Ankalaev']
        print('printing xdf')
        print(xdf)

        #need fighter name, opp name to feed into "loc"
        #need to get the "result" from the fightdb_df.


        #could use merge, then prune? hmmm.
        
        return xdf
    

    #Open database
    #Open predictions in progress
    #Search bets in progress for row where Fighter = Fighter, Opp = Opponent, and Date = Date.
    #Concat bets in progress row with results, ko, sub, etc. Assign to "Completed" Predictions tab.
    #Set graphs to look at "Completed Predictions"
