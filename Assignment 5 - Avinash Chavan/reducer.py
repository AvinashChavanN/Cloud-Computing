#!/usr/bin/python

from operator import itemgetter
import sys
import time
import datetime

current_word = None
current_count = 0
word = None

# input comes from STDIN
for line in sys.stdin:
    # parse the input we got from mapper.py
    word, count = line.split('\t', 1)

    # convert count (currently a string) to int
    try:
        if word != 'time':
            count = int(count)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_word == word:
        current_count += count
    else:
        if word == 'time':
            Time_Before = count
        else:
            # write result to STDOUT
            print '%s\t%s' % (current_word, current_count)
            current_count = count
            current_word = word

# do not forget to output the last word if needed!
if current_word == word:
    print '%s\t%s' % (current_word, current_count)

Time_After = time.time()
Difference_In_Times = Time_After-float(Time_Before)
print 'Time after %s' %Time_After
print 'Time before %s' %float(Time_Before)

Minutes = datetime.datetime.fromtimestamp(Difference_In_Times).minute
Seconds = datetime.datetime.fromtimestamp(Difference_In_Times).second
print 'Time taken to upload : {} min {} sec'.format(Minutes,Seconds)
