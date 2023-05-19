import streamlit as st
import pandas as pd
import fightscraper as fs
import calculator as cal
import model as md
import os
import numpy as np
import plotly.graph_objects as go
import altair as alt

st.set_page_config(layout="wide")


@st.cache_data
def load_df_fromsheet (location, sheetname):
    df = pd.read_excel(location, index_col=0, sheet_name=sheetname)
    return df

@st.cache_data
def load_df (location):
    df = pd.read_excel(location, index_col=0) #---> specifically commands to use first column as index.
    #df = pd.read_excel(location)              #index_col=False can be used to force pandas to not use the first column as the index
    return df

@st.cache_resource
def loadModel(modeldf, yColName, testsize, trainsize, seed1, seed2):
    model = md.generateModel(modeldf, yColName, testsize, trainsize, seed1, seed2)
    return model

@st.cache_data
def generateFighterList(modelDB_df):

    fighterList_df = modelDB_df['Fighter']
    fighterList_df = fighterList_df.drop_duplicates()
    fighterList_df  = fighterList_df.sort_values()
    dfp = pd.DataFrame(['None Selected'])
    fighterList_df = pd.concat([dfp, fighterList_df], axis=0)

    return fighterList_df


class PredictionBundle:
    def __init__(self, WinPrediction, KOPrediction, SUBPrediction):
        self.WinPrediction = WinPrediction
        self.KOPrediction = KOPrediction
        self.SUBPrediction = SUBPrediction


class Bet:

    df = pd.DataFrame()
    

    def __init__(self, Prediction, UnitsStaked, MoneyStaked, MoneyToEarn):
        self.Prediction = Prediction
        self.UnitsStaked = UnitsStaked
        self.MoneyStaked = MoneyStaked
        self.MoneyToEarn = MoneyToEarn

        BetDic = { 'Units Staked' : [self.UnitsStaked],
                'Money Staked' : [self.MoneyStaked],
                'To Earn' : [self.MoneyToEarn],
                }

        self.df = pd.DataFrame(BetDic)
        self.df = pd.concat([self.Prediction.df, self.df], axis = 1, ignore_index=False)
        


def highlight_survived(s):
    return ['background-color: DarkSlateGrey']*len(s) if s['BET RESULT'] == "W" else ['background-color: #330000']*len(s)


class Prediction:
    score = 0
    oneUnitReturn = 0
    MDL_probability = 0
    MDL_odds = 0
    site_probability = 0
    site_odds = 0
    color = ""
    prediction = ""
    
    df = pd.DataFrame()


    def __init__(self, Fighter, Opponent, WeightClass, Date, PredictionType):
        self.Fighter = Fighter
        self.Opponent = Opponent
        self.Date = Date
        self.PredictionType = PredictionType
        self.WeightClass = WeightClass



    def setPrediction(self, MDL_prediction):
        #takes a 1 or 0

        if(self.PredictionType == "WIN"):
            if(MDL_prediction == 1):
                self.prediction = "WIN!"
            if(MDL_prediction == 0):
                self.prediction = "LOSE"
            
        
        if(self.PredictionType == "KO"):
            if(MDL_prediction == 1):
                self.prediction = "KO!"
            if(MDL_prediction == 0):
                self.prediction = "NO KO"        

        if(self.PredictionType == "SUB"):
            if(MDL_prediction == 1):
                self.prediction = "SUB!"
            if(MDL_prediction == 0):
                self.prediction = "NO SUB"

    
    def setVars(self, mdl_odds, site_odds, betunitsize):
        self.score = cal.PredictionScore(mdl_odds, site_odds)
        self.oneUnitReturn = cal.calculateOneUnitReturn(site_odds, betunitsize)
        self.color = cal.GetPredictionScoreColor(self.score)
        


class PredictionDFGenerator():
    def generate(self, prediction):

        predDic = { 'Fighter' : prediction.Fighter,
                        'PREDICTION TYPE' : prediction.PredictionType,
                        'SCORE' : prediction.score,
                        'MDL % CHANCE' : prediction.MDL_probability,
                        'MDL ODDS' : prediction.MDL_odds,
                        'MDL PRED' : prediction.prediction,
                        'SITE ODDS' : prediction.site_odds,
                        'OPP NAME' : prediction.Opponent,
                        'WEIGHT CLASS' : prediction.WeightClass,
                        'Date' : prediction.Date,
                        '1 UNIT RETURN' : prediction.oneUnitReturn,
                        
                        }
        
        df = pd.DataFrame(predDic)
        return(df)
    
    



        




datafolder = 'C:/Users/nkpna/OneDrive/Desktop/Coding/VS Code Projects/UFC project/data/'
scrapedEventsDB_Name = '_t1_scrapedeventsTEST.xlsx'
fightDB_Name = '_t2_fightDBTEST.xlsx'
fighterDB_Name = '_t3_fighterDBdfTEST.xlsx'
modelDB_Name = 'modelDB.xlsx'
allBets_Name = 'allbets.xlsx'


newdf = pd.DataFrame()
testdatestring = 'Oct 27, 1991'
newdate = pd.to_datetime(testdatestring, format ='%b %d, %Y')

#Loading databases
scrapedEvents_df = load_df(datafolder + scrapedEventsDB_Name)
fightDB_df = load_df(datafolder + fightDB_Name)
fighterDB_df = load_df(datafolder + fighterDB_Name)
#fullstatsDB_df = load_df(datafolder+ "fullstats.xlsx")
advancedStatsDB_df = load_df(datafolder+ "advancedStatsDB.xlsx")

modelDB_df = load_df(datafolder + modelDB_Name)

eligiblefighters = generateFighterList(modelDB_df)

#Load betting models.
winModel = loadModel(modelDB_df, "Result", .3, .7, 7727, 234)
KOModel = loadModel(modelDB_df, "KO GIVEN", .3, .7, 787, 12358)
SUBModel = loadModel(modelDB_df, "SUB GIVEN", .3, .7, 189, 357)

        
betunitsize = 10




#Updates database with latest fight data.
scrapeResult = st.button('UPDATE DATABASE')
if scrapeResult == True:
        
        
        newfightdata_df, addedEvents = fs.FullEventsPagesScraper(scrapedEvents_df)
        addedEventsNames = []
        fighternamesList = []
        fighterList = []
        


        if (newfightdata_df.empty == False):
            for event in addedEvents:
                addedEventsNames.append(event.name)

                for fighter in event.fighters:
                        fighternamesList.append(fighter.name)
                        fighterList.append(fighter)

            newfightdata_df.columns = fightDB_df.columns.values
            
            fightDB_df = pd.concat([fightDB_df, newfightdata_df], axis=0)
            fightDB_df.sort_values(by="Date", ascending=False)

            fighterDB_df, fightersAddedToDB = fs.updateFighterDB(fighterDB_df, fighterList)
            scrapedEvents_df = fs.updateEventDB(scrapedEvents_df, addedEvents)
        

            st.write('The following events were scraped and added to the database: ')
            st.write(addedEventsNames)


            st.write('The following fighters were added to the DB:')
            st.write(fightersAddedToDB)


            addedEvents = ""
            fighternamesList = ""

            #Saves post-scrape files to excel
            scrapedEvents_df.to_excel(datafolder + '_t1_scrapedeventsTEST.xlsx')
            fightDB_df.to_excel(datafolder +'_t2_fightDBTEST.xlsx')
            fighterDB_df.to_excel(datafolder + '_t3_fighterDBdfTEST.xlsx')

            
            #Runs calculations on post-scrape files to generate advanced statistics
            advancedStatsDB_df = cal.createAdvStatsdf(pd.to_numeric(fightDB_df), fighterDB_df) #make it so that instead of "updatedfighterDB", it's just fighterDB?
            advancedStatsDB_df.to_excel(datafolder + 'fullstats.xlsx')

        else:
            st.write('No events to add to database.')
            print('No events to add to database')
                



tab1, tab2, tab3 = st.tabs(["Prob Calcs", "Predictions History", "Fight Database"])

#Predictions tab
with tab1:

    secondFormDisabled = True
    st.title = ("UFC Betting model - NK")
    
    if 'betsTBD_df' not in st.session_state:
        st.session_state.betsTBD_df = load_df_fromsheet (datafolder + allBets_Name, "TBD BETS")
        

        if ('Date' in st.session_state.betsTBD_df):
            st.session_state.betsTBD_df['Date']  = pd.to_datetime(st.session_state.betsTBD_df['Date'], utc=False)
            
    if 'betsRESOLVED_df' not in st.session_state:
        st.session_state.betsRESOLVED_df = load_df_fromsheet(datafolder + allBets_Name,'RESOLVED BETS' )

    if 'predBundleList' not in st.session_state:
        st.session_state.predBundleList = []


    with st.form("form1", clear_on_submit=True):
        fighterboxlist = []
        opponentboxlist = []
        winNumInputList = []
        subNumInputList = []
        KONumInputList = []
        wtclassInput = []
        savebuttonslist = []
        predDFgen = PredictionDFGenerator()


        if 'predlist' not in st.session_state:
            st.session_state.predlist = []

        wtclasses = ['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight',
                     'Heavyweight', "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight", "Catch weight"]

        col1a, col2a, col3a, col4a, col5a, col6a = st.columns([1.5,1.5,1,1,1,1]) #number of columns
        with col1a:
            st.write("FIGHTER")

        with col2a:
            st.write("OPPONENT")

        with col3a:
            st.write("WIN ODDS")

        with col4a:
            st.write("KO ODDS")

        with col5a:
            st.write("SUB ODDS")

        with col6a:
            st.write("Weight Class")
        
        

        for j in range(0,5): # # of rows
            col1, col2, col3, col4, col5, col6 = st.columns([1.5,1.5,1,1,1, 1]) #number of columns
            with col1:
                fighterboxlist.append(st.selectbox( label= "fighterbox" + str(j), options = eligiblefighters, label_visibility = "hidden"))
            with col2:
                opponentboxlist.append(st.selectbox( label= "opponentbox" + str(j), options = eligiblefighters, label_visibility = "hidden"))            
            with col3:
                winNumInputList.append(st.number_input(step = 10, value= 0, key= "wininput" + str(j), label='placeholder', label_visibility="hidden"))
            with col4:
                KONumInputList.append(st.number_input(step = 10, value= 0, key= "koinput" + str(j), label='placeholder', label_visibility="hidden"))
            with col5:
                subNumInputList.append(st.number_input(step = 10, value= 0, key= "subinput" + str(j), label='placeholder', label_visibility="hidden"))
            with col6:
                wtclassInput.append(st.selectbox(key= "weightclass" + str(j), label='placeholder', label_visibility="hidden", options=wtclasses ))

        fightdate = st.date_input("Date of fight")
        fightdate64 = np.datetime64(fightdate)
        
        submitted1 = st.form_submit_button("Calculate probabilities")
        if submitted1:

            if st.session_state.predBundleList:
                st.session_state.predBundleList.clear()


            for x in range(5):
                
                if(fighterboxlist[x]!= "None Selected"):
                    #creates Predictions templates
                    winpred = Prediction(fighterboxlist[x], opponentboxlist[x], wtclassInput[x],  fightdate64, "WIN")
                    KOpred = Prediction(fighterboxlist[x], opponentboxlist[x], wtclassInput[x],  fightdate64, "KO")
                    SUBpred = Prediction(fighterboxlist[x], opponentboxlist[x], wtclassInput[x],  fightdate64, "SUB")
                    
                    #creates a df to be fed into respective models
                    winmodelPrePred_df =  cal.createPrePredictiondf(advancedStatsDB_df,fighterboxlist[x], opponentboxlist[x], fightdate, 5, wtclassInput[x])
                    KOmodelPrePred_df = cal.createPrePredictiondf(advancedStatsDB_df,fighterboxlist[x], opponentboxlist[x], fightdate, 5, wtclassInput[x])
                    SUBmodelPrePred_df =  cal.createPrePredictiondf(advancedStatsDB_df,fighterboxlist[x], opponentboxlist[x], fightdate, 5, wtclassInput[x])

                    #calculates predict_proba for each prediction (probability of 1:win for fighter A.)
                    winpred.MDL_probability = winModel.predict_proba(winmodelPrePred_df)[:,1]
                    KOpred.MDL_probability = KOModel.predict_proba(KOmodelPrePred_df)[:,1]
                    SUBpred.MDL_probability = SUBModel.predict_proba(SUBmodelPrePred_df)[:,1]

                    #Uses models and PrePrediction dfs (fighter 1 vs fighter 2 data) to predict 1:win, 0:loss.
                    winpred.setPrediction(winModel.predict(winmodelPrePred_df))
                    KOpred.setPrediction(KOModel.predict(KOmodelPrePred_df))
                    SUBpred.setPrediction(SUBModel.predict(SUBmodelPrePred_df))

                    winpred.MDL_odds = cal.decimalOddsToAmerican(winpred.MDL_probability)
                    KOpred.MDL_odds = cal.decimalOddsToAmerican(KOpred.MDL_probability)
                    SUBpred.MDL_odds = cal.decimalOddsToAmerican(SUBpred.MDL_probability)

                    winpred.site_odds = winNumInputList[x]
                    KOpred.site_odds = KONumInputList[x]
                    SUBpred.site_odds = subNumInputList[x]

                    #Sets Prediction.Score and Prediction.OneUnitReturn
                    winpred.setVars(winpred.MDL_odds, winpred.site_odds, betunitsize)
                    KOpred.setVars(KOpred.MDL_odds, KOpred.site_odds, betunitsize)
                    SUBpred.setVars(SUBpred.MDL_odds, SUBpred.site_odds, betunitsize)

                    #sets Prediction.df to a DF that contains all of the information from the prediction.
                    winpred.df = predDFgen.generate(winpred)
                    KOpred.df = predDFgen.generate(KOpred)
                    SUBpred.df = predDFgen.generate(SUBpred)

                    st.session_state.predBundleList.append(PredictionBundle(winpred, KOpred, SUBpred))

                    secondFormDisabled = False

    y = 0
    with st.expander(label="PREDICTIONS: "):
        
        chkboxlist = []
        buttonlist = []
        predictionstoadd_df = pd.DataFrame()
        predictionsDFList_temp = []
        UnitInputList = []
        DollarsBetList = []
        BetReturnList = []
        predictionsList = []
        
        for predBundle in st.session_state.predBundleList:


            st.write(predBundle.WinPrediction.Fighter + " VS " + predBundle.WinPrediction.Opponent)


            predictionsList.append(predBundle.WinPrediction)
            predictionsList.append(predBundle.KOPrediction)
            predictionsList.append(predBundle.SUBPrediction)

            
            col1, col2, col3 = st.columns(3)

            #COLUMN FOR "WIN" BETS
            with col1:
                cola, colb, colc = st.columns(3)
                                     

                with colb:
                    st.write('WIN BET SUMMARY')
                    st.write(predBundle.WinPrediction.Fighter + " WIN vs " + predBundle.WinPrediction.Opponent)

                st.write('Site odds: ' + str(predBundle.WinPrediction.site_odds))
                st.write('MDL odds: ' + str(predBundle.WinPrediction.MDL_odds))
                st.write('SCORE: ' + str(predBundle.WinPrediction.score))
                st.write('1U return: ' + str(predBundle.WinPrediction.oneUnitReturn))
                st.write('MODEL PREDICTION: ' + predBundle.WinPrediction.prediction)

                st.write('Recommended bet: ' + cal.GetBetRecommendation(predBundle.WinPrediction.score))

                st.write('')
                st.write('')
                st.write('')

                UnitInputList.append(st.number_input(label= "Units to bet: ", key="winbet" + str(y), step=1))
                DollarsBetList.append(round(UnitInputList[y] * betunitsize))
                BetReturnList.append(round(cal.calculateReturn(predBundle.WinPrediction.site_odds, betunitsize, UnitInputList[y])))

                st.write("Win: $" + str(BetReturnList[y]))
                chkboxlist.append(st.checkbox(label="SAVE BET:", key = str(y)))
                y = y + 1  


            #COLUMN FOR "KO" BETS
            with col2:
                cola, colb, colc = st.columns(3)               
                            

                with colb:
                    st.write('KO BET SUMMARY')
                    st.write(predBundle.KOPrediction.Fighter + " KO vs " + predBundle.KOPrediction.Opponent)

                st.write('Site odds: ' + str(predBundle.KOPrediction.site_odds))
                st.write('MDL odds: ' +  str(predBundle.KOPrediction.MDL_odds))
                st.write('SCORE: ' + str(predBundle.KOPrediction.score))
                st.write('1U return: ' + str(predBundle.KOPrediction.oneUnitReturn))
                st.write('MODEL PREDICTION: ' + predBundle.KOPrediction.prediction)
                st.write('Recommended bet: ' + cal.GetBetRecommendation(predBundle.KOPrediction.score))
                st.write('')
                st.write('')
                st.write('')

                UnitInputList.append(st.number_input(label= "Units to bet: ", key="kobet" + str(y), step=1))
                DollarsBetList.append(round(UnitInputList[y] * betunitsize))
                BetReturnList.append(round(cal.calculateReturn(predBundle.KOPrediction.site_odds, betunitsize, UnitInputList[y])))

                st.write("Win: $" + str(BetReturnList[y]))
                chkboxlist.append(st.checkbox(label="SAVE BET:", key = str(y)))

                y = y + 1    
                                


            #COLUMN FOR "SUB" BETS
            with col3:
                cola, colb, colc = st.columns(3)
                              

                with colb:
                    st.write('SUB BET SUMMARY')
                    st.write(predBundle.KOPrediction.Fighter + " SUB vs " + predBundle.KOPrediction.Opponent)

                st.write('Site odds: ' + str(predBundle.SUBPrediction.site_odds))
                st.write('MDL odds: ' + str(predBundle.SUBPrediction.MDL_odds))
                st.write('SCORE: ' + str(predBundle.SUBPrediction.score))
                st.write('1U return: ' + str(predBundle.SUBPrediction.oneUnitReturn))
                st.write('MODEL PREDICTION: ' + predBundle.SUBPrediction.prediction)
                st.write('Recommended bet: ' + cal.GetBetRecommendation(predBundle.SUBPrediction.score))
                st.write('')
                st.write('')
                st.write('')

                UnitInputList.append(st.number_input(label= "Units to bet: ", key="subbet" + str(y), step=1))
                DollarsBetList.append(round(UnitInputList[y] * betunitsize))
                BetReturnList.append(round(cal.calculateReturn(predBundle.SUBPrediction.site_odds, betunitsize, UnitInputList[y])))

                st.write("Win: $" + str(BetReturnList[y]))

                chkboxlist.append(st.checkbox(label="SAVE BET:", key = str(y)))
                y = y + 1  
            
            

            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")            
            
            
        st.write('UNRESOLVED BETS')
        st.dataframe(st.session_state.betsTBD_df)

        saveBets = st.button("Save selected bets.")
        
        if saveBets:
            p = 0
            betsDF = pd.DataFrame()

            for chkbox in chkboxlist:
                
                if(chkbox == True):

                    bet = Bet(predictionsList[p], UnitInputList[p],DollarsBetList[p], BetReturnList[p])
                    st.session_state.betsTBD_df = pd.concat([st.session_state.betsTBD_df, bet.df], axis = 0, ignore_index=True)
                chkbox = False
                
                p = p + 1
            
            st.session_state.betsTBD_df = st.session_state.betsTBD_df.drop_duplicates()
            
        

            st.write('BETS ADDED!')
                   
        
        # create a excel writer object
            with pd.ExcelWriter(datafolder + allBets_Name ) as writer:
            # to store the dataframe in specified sheet
                st.session_state.betsTBD_df.to_excel(writer, sheet_name="TBD BETS", index=True)
                st.session_state.betsRESOLVED_df.to_excel(writer, sheet_name="RESOLVED BETS", index=True)

            

            predictionsList.clear()
            st.session_state.predBundleList.clear()
            st.experimental_rerun() 

        
        clearpredictions = st.button("Clear Predictions Slips")
        if clearpredictions == True:
            st.session_state.predBundleList.clear()


#BET HISTORY TAB
with tab2:
    
    st.write('UNRESOLVED BETS')
    st.dataframe(st.session_state.betsTBD_df)
    st.write('RESOLVED BETS')
    st.dataframe(st.session_state.betsRESOLVED_df.style.apply(highlight_survived, axis=1))


    updateresults = st.button("Update results")
    if updateresults:  

        fightDB_df['Date'] = pd.to_datetime(fightDB_df['Date'])
        newresolvedBets_df = st.session_state.betsTBD_df.merge(advancedStatsDB_df, on=['Fighter','OPP NAME', 'Date' ],
                                                    how='inner')
        

        st.dataframe(st.session_state.betsTBD_df)
        st.dataframe(advancedStatsDB_df)
        st.write('newresolved bets')
        st.dataframe(newresolvedBets_df)

        if newresolvedBets_df.empty ==  False:

            newresolvedBets_df = newresolvedBets_df.loc[:, ['To Earn','Money Staked',  'Units Staked', 'Fighter', 'PREDICTION TYPE', 'OPP NAME', 'MDL % CHANCE', 'MDL ODDS', 'MDL PRED', 'SITE ODDS', 'WEIGHT CLASS', 'Date',
                                    '1 UNIT RETURN', 'SCORE', 'Result', 'KO GIVEN', 'SUB GIVEN', ]]
            
            betsToRemove_df = newresolvedBets_df.loc[:, ['To Earn','Money Staked',  'Units Staked', 'Fighter', 'PREDICTION TYPE', 'OPP NAME', 'MDL % CHANCE', 'MDL ODDS', 'MDL PRED', 'SITE ODDS', 'WEIGHT CLASS', 'Date',
                                    '1 UNIT RETURN', 'SCORE' ]]

            
            st.session_state.betsTBD_df = pd.concat([st.session_state.betsTBD_df, betsToRemove_df, betsToRemove_df]).drop_duplicates(keep=False)
            

            betresultdic = { 'BET RESULT' : [0],
                            'PROFIT' : [0],
                            'CUMULATIVE PROFIT' : [0],}
            
            betresultdf = pd.DataFrame(betresultdic)
            newresolvedBets_df = pd.concat([betresultdf, newresolvedBets_df], axis = 1, ignore_index= False)
            
            newresolvedBets_df['BET RESULT'] = "L"

            newresolvedBets_df.loc[(newresolvedBets_df['Result'] == "W") & (newresolvedBets_df['KO GIVEN'] == 1) & (newresolvedBets_df['PREDICTION TYPE'] == "KO") , 'BET RESULT'] = "W"
            newresolvedBets_df.loc[(newresolvedBets_df['Result'] == "W") & (newresolvedBets_df['SUB GIVEN'] == 1) & (newresolvedBets_df['PREDICTION TYPE'] == "SUB") , 'BET RESULT'] = "W"
            newresolvedBets_df.loc[(newresolvedBets_df['Result'] == "W")  & (newresolvedBets_df['PREDICTION TYPE'] == "WIN") , 'BET RESULT'] = "W"

            newresolvedBets_df.loc[(newresolvedBets_df['BET RESULT'] == "W"), 'PROFIT'] = (newresolvedBets_df['To Earn']).astype(int)
            newresolvedBets_df.loc[(newresolvedBets_df['BET RESULT'] == "L"), 'PROFIT'] = (newresolvedBets_df['Money Staked'] * -1).astype(int)


            st.session_state.betsRESOLVED_df = pd.concat([st.session_state.betsRESOLVED_df, newresolvedBets_df], axis=0, ignore_index=True )
            print('checking duplicate cols')
            print (st.session_state.betsRESOLVED_df.columns[st.session_state.betsRESOLVED_df.columns.duplicated(keep=False)])

            
            st.session_state.betsRESOLVED_df = st.session_state.betsRESOLVED_df.T.drop_duplicates().T
            st.session_state.betsRESOLVED_df = st.session_state.betsRESOLVED_df.sort_values(by='Date',ascending=True)
            st.session_state.betsRESOLVED_df['CUMULATIVE PROFIT'] = st.session_state.betsRESOLVED_df['PROFIT'].cumsum()

            st.dataframe(st.session_state.betsRESOLVED_df.style.apply(highlight_survived, axis=1))

            with pd.ExcelWriter(datafolder + allBets_Name ) as writer:
                st.session_state.betsTBD_df.to_excel(writer, sheet_name="TBD BETS", index=True)
                st.session_state.betsRESOLVED_df.to_excel(writer, sheet_name="RESOLVED BETS", index=True)

            st.experimental_rerun()


        else:
            st.write('No further bets to update.')



    col1, col2 = st.columns(2)

    with col1:
        
        runningProfit = st.session_state.betsRESOLVED_df['CUMULATIVE PROFIT'].values[-1]
        if (runningProfit > 0):
            profitColor = 'green'

        if (runningProfit < 0):
            profitColor = 'red'

        st.write('CUMULATIVE PROFIT')


        html_str = f"""
        <style>
        p.a {{
        color: {profitColor};
        font: bold {50}px sans serif;
        }}
        </style>
        <p class="a">${runningProfit}</p>

        """

        st.markdown(html_str, unsafe_allow_html=True)

        st.line_chart(st.session_state.betsRESOLVED_df, y= "CUMULATIVE PROFIT", x="Date")
        source4 = pd.DataFrame({
            "BetResult": st.session_state.betsRESOLVED_df['BET RESULT'],
            "Predictions": st.session_state.betsRESOLVED_df['PROFIT'],
            "PredictionType": st.session_state.betsRESOLVED_df['PREDICTION TYPE']
        })


        #colors = ['red', 'green']
        bar_chart4 = alt.Chart(source4, title="Correct/Incorrect Predictions by Each Model Type").mark_bar().encode(
            x = "PredictionType:O",
            y = "count(Predictions):Q",
            #color = "BetResult:N"
            color=alt.Color("BetResult:N",
                    scale=alt.Scale(
                        range=['#330000', 'DarkSlateGrey']))

        )

        st.altair_chart(bar_chart4, use_container_width=True)






        testgb = st.session_state.betsRESOLVED_df.groupby(['PREDICTION TYPE'], as_index=False).sum()
        bar_chart5 = alt.Chart(testgb, title="Profit by Each Model Type").mark_bar().encode(
            x = "PREDICTION TYPE:O",
            y = "PROFIT:Q",
            color=alt.condition(
            alt.datum.PROFIT < 0,  
            alt.value('#330000'),    
            alt.value('DarkSlateGray')  
    )


        )

        st.altair_chart(bar_chart5, use_container_width=True)



    with col2:

        st.write('')
        st.write('')

        testdf = st.session_state.betsRESOLVED_df.loc[:, ['PREDICTION TYPE','BET RESULT', ]]
        freqtable_df = pd.crosstab(testdf['PREDICTION TYPE'], testdf['BET RESULT'])
        
    

#DATA VISUALIZATION
with tab3:

    yaxis = st.selectbox('Y axis', options=advancedStatsDB_df.columns, index=6)
    xaxis = st.selectbox('X axis', options= advancedStatsDB_df.columns, index=20)

    yaggregateType = st.radio('Select Y-Axis aggregate', options = ['sum', 'count', 'mean'])

    st.multiselect('Select graph filters', options = advancedStatsDB_df.columns )

    graphtype = st.radio('Select graph type', options = ['BAR GRAPH', 'LINE CHART'])


    df_filtered = advancedStatsDB_df[advancedStatsDB_df[yaxis] > 0]



    source6 = pd.DataFrame({
    "BetResult": st.session_state.betsRESOLVED_df['BET RESULT'],
    "Predictions": st.session_state.betsRESOLVED_df['PROFIT'],
    "PredictionType": st.session_state.betsRESOLVED_df['PREDICTION TYPE']
    })


    bar_chart7 = alt.Chart(advancedStatsDB_df, title=yaxis +" and " + xaxis).mark_line().encode(
        alt.X(xaxis, sort='-y'),
        y = yaggregateType+"("+yaxis+")",
    ).interactive()

        
    interval = alt.selection_interval()

    bar_chart6 = alt.Chart(df_filtered, title=yaxis +" and " + xaxis).mark_bar().encode(
        alt.X(xaxis, sort='-y'),
        y = yaggregateType+"("+yaxis+")",
        color=alt.condition(interval, xaxis, alt.value('lightgray'))
        ).add_selection(interval)
        

    bar_chart8 = alt.Chart(df_filtered, title=yaxis +" and " + xaxis).mark_bar().encode(
        x = xaxis,
        y = yaxis,
        color=alt.condition(interval, xaxis, alt.value('lightgray'))
        ).interactive()



    bar_chart9 = alt.Chart(df_filtered, title=yaxis +" and " + xaxis).mark_bar().encode(
        x = xaxis,
        y = yaxis,
        color=alt.condition(interval, xaxis, alt.value('lightgray'))
        ).interactive()



    st.altair_chart(bar_chart6, use_container_width=True)
    st.altair_chart(bar_chart7, use_container_width=True)
    st.altair_chart(bar_chart9, use_container_width=True)
