
from asyncio.windows_events import NULL
from pickle import TRUE
from select import select
from unittest import result
import bs4
import requests
import openpyxl
import webbrowser
import time
import pandas as pd
import os
import math
from datetime import datetime
from datetime import date


class Fighter:
    def __init__(self, name, link):
        self.name = name
        self.link = link

class Event:
    def __init__(self, name, link, date, fighters):
        self.name = name
        self.link = link
        self.fighters = fighters
        self.date = date

    
def scrapeFighterData(link):
    #This function scrapes data from a fighter's bio page[link], creates DF with fighter info (HT, DOB, REACH, etc)

    time.sleep(.5)
    fighterpage = requests.get(link)
    fighterpage.raise_for_status()
    fighterpageSoup = bs4.BeautifulSoup(fighterpage.text, 'html.parser')
    fighterStats = []
    global fighterinfodf

    fighterName = fighterpageSoup.find('h2', class_='b-content__title').contents[1].getText().strip()
    boxlist = fighterpageSoup.find('ul', class_='b-list__box-list')
    points = boxlist.findAll('li')

    fighterStats.append(fighterName)

    y= 0

    for point in points:
        data = point.contents[2].getText().strip()
        fighterStats.append(data)


    fighterWeight = fighterStats[2]
    fighterStance = fighterStats[4]
    fighterReach = fighterStats[3]
    fighterDOB = fighterStats[5]
    fighterHeight = fighterStats[1]

    #Creating dictionary
    fighterdata = {'Link': [link],
        'Fighter': [fighterName],
        'HT':[fighterHeight],
        'REACH' : [fighterReach],
        'STANCE' : [fighterStance],
        'WEIGHT' : [fighterWeight],
        'DOB' : [fighterDOB],
        }

    #Creating a DF from the *dictionary above.
    fighterdf = pd.DataFrame(fighterdata)

    return fighterdf

def splitText (iClass):
    wordsList = ["Method:", "Time:", "Time format:", "Referee:", "Details:", "Round:"]
    text = iClass.getText()
    for word in wordsList:
        if word in text:
            text = text.replace(word, '')
            

    return text.strip()


def FullEventsPagesScraper(scrapedEvents_df):
    #This function navigates to the UFC events page and scrapes the data from each event. 
    #Takes a dataframe to see what events have been scraped.

    FullEventsPage = requests.get('http://ufcstats.com/statistics/events/completed?page=all')
    FullEventsPage.raise_for_status()
    FullEventsPageSoup = bs4.BeautifulSoup(FullEventsPage.text, 'html.parser')
    AllEventsLinks = FullEventsPageSoup.select('tr a')
    AllEventsDates = FullEventsPageSoup.select('tr span')
    eventsAdded_list = []

    # print('all events dates')
    # print(AllEventsDates)

    combinedEventsAdded_df = pd.DataFrame()

    i= 0
    for event in AllEventsLinks:

        eventdatetime = datetime.strptime(AllEventsDates[i].getText().strip(), "%B %d, %Y")
        eventdate = eventdatetime.date()

        i = i + 1
        #CHECKS IF CURRENT EVENT HAS ACTUALLY HAPPENED OR IS HAPPENING TODAY.
        if(eventdate >= date.today()):
             print('This event is happening today or in the future, and will not be added in the scrape')
             print(event)
             continue
             
        #Checks if event is already in the database.
        if scrapedEvents_df['EVENT'].str.contains(event.getText().strip()).any() == False:
                    newevent_df, _event = ScrapeEvent(event['href'])
                    eventsAdded_list.append(_event) #list of type Event with all added events.
                    combinedEventsAdded_df = pd.concat([combinedEventsAdded_df, newevent_df], axis = 0, ignore_index=True) #
                    


    print('FULL EVENTS PAGE SCRAPER FINISHED')

    #returns a df with all of the added events dfs combined, and a list of all the Events added.
    return combinedEventsAdded_df, eventsAdded_list



def ScrapeEvent(EventLink):
    #This function takes the link for an event and scrapes that individual event.
    #Returns: df of scraped event, list of Fighter objects of fighters who are in event.

    FightsList = requests.get(EventLink)
    FightsList.raise_for_status()
    FightsListSoup = bs4.BeautifulSoup(FightsList.text, 'html.parser')
    FightsLinks = FightsListSoup.select('tbody tr')


    #if test.getText() == 
    #THIS LOOKS AT 'A' HTML ELEMENTS (LINKS) ASSOCIATED WITH FIGHTER NAME (SO THE LINK ON THE FIGHTER NAME)
    fighterslinks = FightsListSoup.findAll('a', class_= 'b-link b-link_style_black')
    data = FightsListSoup.findAll('li', class_ = 'b-list__box-list-item')


    EVENTLINK = EventLink
    EVENTNAME = FightsListSoup.find('span', class_ = 'b-content__title-highlight').getText().strip()
    DATE = data[0].getText()
    LOCATION = data[1].getText()

  


    DateString = "Date:"
    LocationString = "Location:"

    #scrubbing DATE and Location data
    if(DateString in DATE):
        DATE = DATE.replace(DateString, '')
        DATE = DATE.strip()

    if(LocationString in LOCATION):
        LOCATION = LOCATION.replace(LocationString, '')
        LOCATION = LOCATION.strip()
            
    eventFightDataCombined_df = pd.DataFrame()
    

    print("date: " + DATE)
    print("location: " + LOCATION)

#SCRAPING EACH INDIVIDUAL FIGHT IN THE EVENT
    for x in FightsLinks:
        print(x['data-link'])
        fightlink = x['data-link'] #link is a string
        fightdf = scrapefight(fightlink)
        eventFightDataCombined_df = pd.concat([eventFightDataCombined_df, fightdf], axis = 0)


    eventFightDataCombined_df['Event Link'] = EVENTLINK
    eventFightDataCombined_df['Event'] = EVENTNAME
    eventFightDataCombined_df['Location'] = LOCATION
    eventFightDataCombined_df['Date'] = DATE


    fightersList = []

    #THIS LOOKS THROUGH EVERY 'A' (LINK), SETS FIGHTER NAME TO THE TEXT, AND FIGHTER BIOLINK TO THE LINK.
    #THEN WE CREATE A "FIGHTER" AND ADD THIS FIGHTER TO FIGHTERSLIST

    for link in fighterslinks:
        _fightername = link.getText().strip()
        _fighterbiolink = link.get('href').strip()
        fighter = Fighter(_fightername, _fighterbiolink)
        fightersList.append(fighter)

        event = Event(EVENTNAME, EVENTLINK, DATE, fightersList)
        
    
    return eventFightDataCombined_df, event

   

        

def scrapefight(link):
    #This function takes a link for a fight and scrapes the statistics from that individual fight.
    #returns a df


    time.sleep(.5)

    fighter1Data = []
    fighter2Data = []

    ufcpage = requests.get(link)
    ufcpage.raise_for_status()
    ufcpageSoup = bs4.BeautifulSoup(ufcpage.text, 'html.parser')


    def splitOfText(_string):
        splitList = _string.getText().strip().split()
        recombinedStringList = [''] * 2
        recombinedStringList[0] = splitList[0]
        recombinedStringList[1] = splitList[2]
        return recombinedStringList

    
    sharedData = ufcpageSoup.findAll('i', class_ = 'b-fight-details__text-item')
    _details = ufcpageSoup.findAll('p', class_ = 'b-fight-details__text')
    resultdetails = ufcpageSoup.findAll('div', class_='b-fight-details__person')

    

    WEIGHTCLASS = ufcpageSoup.find('i', class_ ='b-fight-details__fight-title').getText().strip()
    METHODOFVICTORY = ufcpageSoup.find('i', style ='font-style: normal').getText().strip()
    ROUND = splitText(sharedData[0])
    TIME = splitText(sharedData[1])
    TIMEFORMAT = splitText(sharedData[2])
    REFEREE = splitText(sharedData[3])
    DETAILS = splitText(_details[1])
    RESULTS = []
    FIGHTERS = []

    for x in resultdetails:
        FIGHTERS.append(x.find('h3').getText().strip())
        RESULTS.append(x.find('i').getText().strip())


    print ("Round: " + ROUND)
    print ("Time: " + TIME)
    print ("Time Format: " + TIMEFORMAT)
    print ("Referee: " + REFEREE)
    print ("DETAILS: " + DETAILS)
    

    a= -1
    b= -1


    if (ufcpageSoup.findAll('table')): 
        Table1= ufcpageSoup.findAll('table')[0]
        columns1 = Table1.findAll("td")

        Table2= ufcpageSoup.findAll('table')[2]
        columns2 = Table2.findAll("td")

        for column in columns1: #for each item in the columns list, each 'td'.
            a = a + 1
            
            T1F1Data= column.findAll('p')[0] #search the item (column) for the first 'p'
            T1F2Data = column.findAll('p')[1]

            if(a==2 or a==4 or a==5):
                fighter1Data.extend(splitOfText(T1F1Data))
                fighter2Data.extend(splitOfText(T1F2Data))
                continue
        
            fighter1Data.append(T1F1Data.getText().strip())
            fighter2Data.append(T1F2Data.getText().strip())


        for column in columns2: #for each item in the columns list. 
            b = b + 1
            T2F1Data= column.findAll('p')[0] #search the item (column) for the second 'p'
            T2F2Data = column.findAll('p')[1] 

            if(b==1 or b>=3):
                fighter1Data.extend(splitOfText(T2F1Data))
                fighter2Data.extend(splitOfText(T2F2Data))
                continue
        
            fighter1Data.append(T2F1Data.getText().strip())
            fighter2Data.append(T2F2Data.getText().strip())
            




    fighter1DataCopy = fighter1Data.copy()
    fighter1Data.extend(fighter2Data)
    fighter2Data.extend(fighter1DataCopy)
    

   #1. make a DF template for f1 and f2.


    fighter1Dic = {'Fight Link': [link],
                'Event Link' : [0],
                'Event' : [0],
                'Location': [0],
                'Date':[0],
                'Weight Class': [WEIGHTCLASS],
                'Result':[RESULTS[0]],
                'Result Details' : [METHODOFVICTORY],
                'Ending Round' : [ROUND],
                'Ending Time' : [TIME],
                'Time Format' : [TIMEFORMAT],
                'Referee' : [REFEREE] ,
                'Details' : [DETAILS]
                }
    
    fighter2Dic = {'Fight Link': [link],
                'Event Link' : [0],
                'Event' : [0],
                'Location': [0],
                'Date':[0],
                'Weight Class': [WEIGHTCLASS],
                'Result':[RESULTS[1]],
                'Result Details' : [METHODOFVICTORY],
                'Ending Round' : [ROUND],
                'Ending Time' : [TIME],
                'Time Format' : [TIMEFORMAT],
                'Referee' : [REFEREE] ,
                'Details' : [DETAILS]
                }
    


    #2. assign values to DF of f1 and f2.
    #3. add common data to f1 and f2



    fighter1df = pd.DataFrame(fighter1Dic)
    fighter2df = pd.DataFrame(fighter2Dic)


    #3.5: add rest of data.

    #turning horizontal list of data into dataframe
    a = pd.DataFrame(fighter1Data).T
    b = pd.DataFrame(fighter2Data).T


    df1combined = pd.concat([fighter1df, a], axis = 1)
    df2combined = pd.concat([fighter2df, b], axis = 1)



    #4 combine f1 and f2 vertically

    fightData_df = pd.concat([df1combined, df2combined])


    #5 return combined df


    #Returns a DF with two rows. Fighter A vs Fighter B, and Fighter B vs Fighter A.
    return(fightData_df)



def updateFighterDB(fighterDB_df, fighterlist):
    #Updates the database[fighterDB_df], with fighters in fighterlist.

    addedFightersNames_list = []

    fighterDB_df.reset_index(drop=True)

    for fighter in fighterlist:
        if fighterDB_df['Fighter'].str.contains(fighter.name).any() == False:
            fighterdf = scrapeFighterData(fighter.link)
            addedFightersNames_list.append(fighter.name)
            fighterDB_df = pd.concat([fighterDB_df, fighterdf], axis = 0, ignore_index=True )

    #Returns updated fighterDB_df and a list of fighters added.
    return fighterDB_df, addedFightersNames_list

                    

def updateEventDB(scrapedeventsdb_df, eventlist):
    #Takes current scraped events DB and updates it with events from list eventlist

    scrapedeventsdb_df.reset_index(drop=True)
    eventnames = []
    eventlinks = []
    eventdates = []

    for event in eventlist:
            eventnames.append(event.name)
            eventlinks.append(event.link)
            eventdates.append(event.date)

    combinedEventDic = { 'EVENT' : eventnames,
                'EVENT LINK' : eventlinks,
                'DATE' : eventdates
                }

    combinedEventdf = pd.DataFrame(combinedEventDic)

    scrapedeventsdb_df = pd.concat([scrapedeventsdb_df, combinedEventdf] , axis = 0, ignore_index=True)

    #Returns scrapedeventsdb_df, with new updates added.
    return scrapedeventsdb_df         
    
    




