# ELO_multiplayer_match
Tools and model to predict multi-player match result based on ELO rating.


One way to predict the result of a Team Match of **Chess.com** is using the **predict.py** script in this respository. The notebook **predict_tutorial.ipynb** shows how to use this script to make a prediction.

Other ways to predict the result is running notebooks **MatchResutlEstimation.ipynb** or **MatchVarianceAnalysis.ipynb**. The first one detail the procedure to predict a Match of two teams, the second one is a more complete notebook to predict, where is considered a bias as a variable in the Match.

## Running from Terminal
The default way to execute Match Result Prediction using **predict.py** is using *Terminal*.

1. The first step is clonning this repository into the folder you want:

    `$ git clone https://github.com/francof2a/ELO_multiplayer_match.git`


2. You need information about the match over you want to get a prediction:
    * **API url**: get the Chess.com url of the match. For example https://api.chess.com/pub/match/995756 is the match **WL2019 R4: Team Argentina vs Team England**.
    * **ID**: get the ID (identifier of the match) which is the las part of the url. For the last example, the ID is 995756.
    * **chess.com url**: if you only get the match url of chess.com, for the last examplie it would be https://www.chess.com/club/matches/team-argentina/995756, you can get the ID (last field of the url), and build the API url concatenating _https://api.chess.com/pub/match/_ and _ID_, or just use ID.


3. Run the prediction just doing next:

    `$ python predict.py -id "995756"`

**Note**: The first time you run a prediction for a match, all the info is downloaded from _chess.com API_ and it will take several minutes depending of connection, so please be patient. A backup of that information is stored in _data_ folder to avoid repeat downloading.

A report like this will be printed in terminal:

```
Match info:
	Name:	WL2019 R4: Team Argentina vs Team England
	Team A:	Team Argentina
	Team B:	Team England

Reading ELOs list
	Loading from web ...
	Done!
	Saving backup file./data/wl2019-r4-team-argentina-vs-team-england_match_stats.xlsx

Simulation of match - Result prediction:
Team A (Team Argentina):
	Win chances = 100.00 %
	Draw chances = 0.00 %
	Lose chances = 0.00 %
Expected final score = 423.88 (±11.16) - 244.12 (±11.16)
Expected effectiveness = 63.45 % - 36.55 %

Calculation of Variance over Team A ELOs

Done!
```

All the plots generated will be stored in **outputs** folder.

## Running from jupyter
The **predict.py** script can be executed from a jupyter notebook (like this) emulating console/terminal command line entry using **!**:

`!python predict.py -id "995756"`

All the plots generated will be stored in **outputs** folder.

If you have a **ipython console** or you are running the script in a jupyter notebook, you can show the plots inline adding **-plot** argument, then:

`%matplotlib inline`

`%run predict.py -id "995756" -plot`


## Arguments
Next, the list of arguments supported by **predict.py**, and examples:

   * -h : help about arguments.
   * -id "< match_id >" : (int) ID assigned to the match by chess.com.
   * -url "< API match URL >" : (str) URL of the chess.com API assigned to the match. Don't use if you have already especified ID.
   * -N < number of trials > : (int) number of trials to execute during prediction. Default value = 1000.
   * -Nb < number of trials > : (int) number of trials to execute during prediction considering bias for a team. Default value = 1000.
   * -bias < ELO bias > : (float) ELO bias value (offset) for Team A (first team of the match). Default value = 0. It is necessary specify this value to enable _biased analysis_.
   * -plot : enable inline plot in _ipython_ console or jupyter notebook.
   * -u : force update match data from the web. It is not necessary use this for the first prediction of a particular match.

Examples:

`$ python predict.py -h"`

`$ python predict.py -id "995756"`

`$ python predict.py -url "https://api.chess.com/pub/match/995756"`

`$ python predict.py -id "995756" -N 500`

`$ python predict.py -id "995756" -bias 21.5`

`$ python predict.py -id "995756" -bias 21.5 -Nb 750`

`$ ipython predict.py -id "995756" -plot`

`$ python predict.py -id "995756" -u`