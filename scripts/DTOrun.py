""" 
Tool to automatically run the DTOcean WP4:Moorings and Foundations tool multiple times while automatically changing the input parameters file.
Jon Hardwick, University of Exeter
05-2016
V1.0
"""

import os
import sys
import datetime
import subprocess

import numpy as np


def DTOread(filename_in, *args): #Function to read all the data from the initial input file 
    Var = [0] * len(open(filename_in).readlines())
    if len(args) == 0:
        filename_out = filename_out
    else:
        filename_out = args[0]
    print('Opening file...')
    fid = open(filename_in)
    n=0
    fid2 = open(filename_out,'w')
    print('Done\nReading initial lines...')
    for i in range(0,22): # Writes the top lines of the input file into the new file without making any changes
        A = fid.readline()
        fid2.write(A)
    print('Done\nReading data...')
    with open(filename_in) as openfileobject: #Reads all of the data values into the Variable 'Var'
        for line in openfileobject:
            A = fid.readline()
            Var[n] = A
            n=n+1
    Varind = np.full((len(Var),1), False, dtype=bool) # Identifies which lines from the input file contain variables which can be altered. 
    print('Done\nIdentifying variables...')

    for i in range(0,len(Var)):    
        if len(Var[i]) > 33:
            if ((Var[i][0:32] == '                                ') & (Var[i][33] != '#') & (Var[i][33] != ' ')):
                Varind[i] = True
            else:
                Varind[i] = False
    print('Done\n\n')
    return Var,Varind
    
def DTOwrite(filename_out,Var): # Function to write the new variables after editing to the output file. 

    fid2 = open(filename_out,'a');
    print('Writing to file ...');
    for i in range(0,len(Var)):
        fid2.write(Var[i]);
    print('Done\n')
    
def DTOchanges(Var,Varind,changes,count): #Function to make the changes to the variables as instructed by that values in the changes file.
    
    print('******************************************\n')
    print('****Altering Variables and running tool***\n')
    print('******************************************\n\n')
    thirtytwo = '                                '
    
    for i in range(0,len(changes)-1,2):
        q = changes[i]
        for k in range(0,len(Var)):
            if ~Varind[k]:
                continue
            if q.lower() in Var[k].lower():
                sind = Var[k].find(', #')
                print('Variable to change:',Var[k][sind+3:-1])
                print('Current Value:',Var[k][32:sind])
                newv = changes[i+1]
                print('New Value: ',newv)
                Var[k] = thirtytwo + str(newv) + ', #' + Var[k][sind+3:-1] + '\n'
                Varcheck = True
        if Varcheck == False:
                print('Error: ' + q + ' does not match any variable. Skipping.\n')
    return Var

def main(changesfile):

    fpath = os.path.dirname(os.path.realpath(__file__)) #Path to folder containing winmake.bat
    filename_in = os.path.join(fpath, "..", 'examples', 'shetland.py') #Name of original test_main.py file
    filename_out = os.path.join(fpath, "..", 'examples', 'shetland_altered.py') #Name of new test_main.py file
#    changesfile = os.path.join(fpath, "..", 'scripts', 'alterations.txt') #Name of text file containing changes.
    logfile = os.path.join(fpath, "..", 'dtocean-moorings.log') #Path to the dtocean-moorings log file
    it = len(open(changesfile).readlines())
    fid = open(changesfile, 'r') 
    
    print('                           #####')
    print('                       #######')
    print('            ######    ########       #####')
    print('        ###########/#####\#####  #############')
    print('    ############/##########--#####################')
    print('  ####         ######################          #####')
    print(' ##          ####      ##########/@@              ###')
    print('#          ####        ,-.##/`.#\#####               ##')
    print('          ###         /  |$/  |,-. ####                 #')
    print('         ##           \_, $\_, |  \  ###')
    print('         #              \_$$$$$`._/   ##')
    print('                          $$$$$_/     ##')
    print('                          $$$$$        #')
    print('                          $$$$$')
    print('                          $$$$$')
    print('                          $$$$$')
    print('                          $$$$$')
    print('                         $$$$$')
    print('                         $$$$$')
    print('                         $$$$$')
    print('                         $$$$$        ___')
    print('                         $$$$$    _.-/   `-._')
    print('                        $$$$$   ,/           `.')
    print('                        $$$$$  /               \ ')
    print('~~~~~~~~~~~~~~~~~~~~~~~$$$$$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('   ~      ~  ~    ~  ~ $$$$$  ~   ~       ~          ~')
    print('       ~ ~      .o,    $$$$$     ~    ~  ~        ~')
    print('  ~            ~ ^   ~ $$$$$~        ______    ~        ~')
    print('_______________________$$$$$________|\\\\\\\_________________')
    print('                       $$$$$        |>\\\\\\\ ')
    print('    ______             $$$$$        |>>\\\\\\\ ')
    print('   \Q%=/\,\            $$$$$       /\>>|#####| ')
    print('    `------`           $$$$$      /=|\>|#####|')
    print('                       $$$$$        ||\|#####|')
    print('                      $$$$$$$          ||"""||')
    print('                      $$$$$$$          ||   ||')
    print('                     $$$$$$$$$')
    print('"""""""""""""""""""""$$$$$$$$$"""""""""""""""""""""""""""""""')
    print('*********DTOcean WP4 module validation test program**********')
    print('*******************Enjoy your validation!********************')
    
    fid_log = open(logfile,'a')
    fid_log.write('                         | \n')
    fid_log.write('                     \       / \n')
    fid_log.write('                       .-"-. \n')
    fid_log.write('                  --  /     \  -- \n')
    fid_log.write(' `~~^~^~^~^~^~^~^~^~^-=======-~^~^~^~~^~^~^~^~^~^~^~` \n')
    fid_log.write(' `~^_~^~^~-~^_~^~^_~-=========- -~^~^~^-~^~^_~^~^~^~` \n')
    fid_log.write(' `~^~-~~^~^~-^~^_~^~~ -=====- ~^~^~-~^~_~^~^~~^~-~^~` \n')
    fid_log.write(' `~^~^~-~^~~^~-~^~~-~^~^~-~^~~^-~^~^~^-~^~^~^~^~~^~-` \n')
    fid_log.write(' Testing started at: {} \n\n'.format(datetime.datetime.now()))
    fid_log.close()
    
    
    for count in range(0,it):
        changes = fid.readline()
        changes = changes.strip()
        changes = str.split(changes, ';')
        Var,Varind = DTOread(filename_in,filename_out)
        Var = DTOchanges(Var,Varind,changes,count)
        DTOwrite(filename_out,Var)
        fid_log = open(logfile,'a')
        fid_log.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        fid_log.write('Scenario : {}\n'.format(count))
        fid_log.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
        fid_log.close()
        subprocess.call(['python',filename_out])
        fid_log = open(logfile,'a')
        fid_log.write('Run complete\n\n\n\n')
        fid_log.close()

# Enter alteration.txt path when calling this script, i.e.:
# python DTOrun.py ../tests/alterations.py
if __name__ == "__main__":        
        
    main(sys.argv[1])
        