You can use eBird to get lists of plausible species for a specific location and date. Have a look at the API requests - each request results in a list of species with corresponding frequency.
Frequencies are based on eBird checklists and range from 0 to 100 representing the amount of checklists that a species was on in percent.

To generate a whitelist that you can use with the baseline system, simply run the script 'species_whitelist.py' and copy the output into the 'config.py'.
