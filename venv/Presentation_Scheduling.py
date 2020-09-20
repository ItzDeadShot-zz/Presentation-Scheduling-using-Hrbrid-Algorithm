"""CPT244_140659_142413_P01.py
    Created by Aaron at 14-Jun-20"""
import csv
import copy
import random
import math
import time
import os
from prettytable import PrettyTable

# Presentation class to be used as object in grouping presentation group, staffs and timeslot
class Presentation:
    presentation = None
    def __init__(self, name, staff):
        self.name = name
        self.staff = staff
    @staticmethod
    def searchPreset(name):
        for i in range(len(Presentation.presentation)):
            if Presentation.presentation[i].name == name:
                return i
        return -1
    @staticmethod
    def searchStaff(name):
        for i in range(len(Presentation.presentation)):
            if name in Presentation.presentation[i].staff:
                return i
        return -1
    def __repr__(self):
        return "Presentation: {} by Staff1: {}, Staff2: {}, Staff3: {}".format(self.name, self.staff[0], self.staff[1], self.staff[2])

def genetic_algorithm(generation_limit, population_number):
    """run genetic algorithm function"""
    # initialization phase
    generation = 0  #initialize generation
    crossover_prob, mutation_prob = 0.8, 0.7     # initialize probability
    population = initialize_population(population_number)   # initialize population

    # evaluation process (making the first one outside so it wont generate twice)
    population_score = evaluate(population)  # evaluate population by giving each chromosome a score

    # generation phase
    while generation < generation_limit:
        next_population = []
        # take 90% of result from parent with children for next population
        while len(next_population) < int(len(population) * (1.0 - (10 / 100))):
            # selection of parents chromosome process
            parent1_ind, parent2_ind = selection(population_score)
            parent_population = [population[parent1_ind-1], population[parent2_ind-1]]

            # genetic process
            child_population = copy.deepcopy(parent_population)    # assign parent as child
            if random.uniform(0, 1) <= crossover_prob:  # probability having crossover then crossover
                child_population = crossover(parent1_ind-1, parent2_ind-1, population)
            if random.uniform(0, 1) <= mutation_prob:   # probability having mutation then mutation happen
                mutation(child_population[0])
                mutation(child_population[1])

            # evaluation process
            parent_score = evaluate(parent_population)
            child_score = evaluate(child_population)

            # selection process
            better_population, better_score = parent_population, parent_score   # assign parent as the best 2 result first
            if better_score[0][1] > better_score[1][1]:   # arrange better score in ascending order
                better_population = [parent_population[1], parent_population[0]]
                better_score = [parent_score[1], parent_score[0]]
            for ind1, score in child_score: # traverse and check each child
                for ind2, better in enumerate(better_score):    # traverse and check each current result
                    _, score2 = better
                    if score < score2:  # if child penalty score is lower than current result score then change the score and insert child
                        better_population.insert(ind2, child_population[ind1-1])
                        better_score.insert(ind2, (ind1, score))
                        better_population.pop()
                        better_score.pop()
                        break
            annealed_ind1 = simulated_annealing(better_population[0])
            annealed_ind2 = simulated_annealing(better_population[1])
            next_population.append(annealed_ind1)
            next_population.append(annealed_ind2)
            # next_population.append(better_population[0])
            # next_population.append(better_population[1])

        # sorting process
        population_score.sort(key=lambda tup: tup[1])   # sorting population in ascending order according to scores
        population = [population[chromosome[0] - 1] for chromosome in population_score]

        # add top 10% of original population for next population
        j = int(len(population) * 10 / 100)
        for i in range(j):
            next_population.append(population[i])
        population = next_population
        population_score = evaluate(population)

        print("Generation: {0}  Best chromosome: {1}".format(generation, min(population_score, key=lambda tup: tup[1])))
        generation += 1

    best_score = min(population_score, key=lambda tup: tup[1])
    print("Best chromosome after {0} generations: {1}".format(generation_limit, best_score))
    best_solution = population[best_score[0]-1]
    return best_solution

def initialize_population(size):
    """generate population function"""
    population = [] # to store population
    for i in range(size):
        chromosome = [] # to store chromosome
        timeslot = list(range(1, 301))  # to random timeslot of 1-300
        random.shuffle(timeslot)
        for x in staff_to_presentation: # traverse each presentation giving each one timeslot
            chromosome.append([x, timeslot.pop()])
        population.append(chromosome)
    return population

def selection(population_score):
    """select 2 parents chromosome function"""
    selectionlist = population_score[:]    # shallow copy
    random.shuffle(selectionlist)
    result = () # to store chosen parents chromosome
    for _ in range(2):  # 2 times traverse to choose 2 parents
        c1, val1 = selectionlist.pop()  # parent chromosome 1st candidate
        c2, val2 = selectionlist.pop()  # parent chromosome 2nd candidate
        result += (c2,) if val1 > val2 else (c1,)   # make comparison between both parent chromosome candidate and choose the one with lower penalty score
    return result

def evaluate(population):
    """evaluate population function"""
    result = [] # store all chromosome's score
    for x in range(len(population)):    # traverse and check each chromosome
        # creation of new data structure to each chromosome for SCs
        staff_timeslot = {x + 1: [] for x in range(47)}  # to store each staff's schedule timeslot
        for group in population[x]:  # traverse and check each group
            presentation, timeslot = group
            staff = presentation.staff
            for each in staff:  # traverse and check each staff
                staff_timeslot[each] = staff_timeslot[each] + [timeslot]
        #check HCs and SCs
        score = hc02(staff_timeslot)    # check staff with concurrent presentation
        score += hc03(population[x])    # check current selected timeslot with venue unavailability
        score += hc04(population[x])    # check current selected timeslot with staff unavailability
        score += sc01(staff_timeslot)   # check staff with consecutive presentation
        score += sc02(staff_timeslot)   # check staff with number of days
        score += sc03(staff_timeslot)   # check staff with change of venue
        result.append((x+1, score))
    return result

def hc02(staff_timeslot):
    """check staff with concurrent presentation"""
    score = 0
    for _, timeslot in staff_timeslot.items():  # traverse and check each staff
        checked = []    # to store timeslot that has been checked
        for each in timeslot:   # traverse and check each timeslot
            if each not in checked: # if timeslot has not yet checked yet then check it
                checked.append(each)    # record this timeslot as checked
                primary = each % 15
                if primary == 0:
                    primary = 15
                base = each // 61
                # to store all possible concurrent timeslot according to current timeslot
                concurrent_timeslot_list = [primary + 60 * base + 15 * x for x in range(4)]
                for concurrent_timeslot in concurrent_timeslot_list:   # checking each possible concurrent timeslot
                    if concurrent_timeslot in timeslot and concurrent_timeslot != each and concurrent_timeslot not in checked:  # if concurrent timeslot exist then give penalty
                        score += 1000
                    checked.append(concurrent_timeslot)
    return score

def hc03(chromosome):
    """check selected timeslot with venue unavailability function"""
    score = 0
    for _, timeslot in chromosome:    # traverse and check each chromosome
        if timeslot in hc03_csv:    # if current selected timeslot clash with venue unavailability then give penalty
            score += 1000
    return score

def hc04(chromosome):
    """check selected timeslot with group's staffs unavailability function"""
    score = 0
    for presentation, timeslot in chromosome:    # traverse and check each chromosome
        for staff in presentation.staff:    # traverse and check each staff of the group
            if timeslot in hc04_csv[staff-1]:  # if current selected timeslot clash with any staff unavailability then give penalty
                score += 1000
    return score

def sc01(staff_timeslot):
    """check staff with consecutive presentation"""
    score = 0
    for _, timeslot in staff_timeslot.items():  # traverse and check each staff
        checked = []  # to store timeslot that has been checked
        for each in sorted(timeslot):  # traverse and check each timeslot
            if each not in checked:  # if the timeslot hasn't been check yet
                primary = each % 15
                if primary == 0:
                    primary = 15
                    checked.append(each)    # record this timeslot as checked
                if primary >= 12:  # starting from 12th time period, there wont be breaking the staff preference limit
                    continue
                base = each // 61
                n = 1  # increment to find consecutive presentation
                while True:
                    # to store all possible next consecutive presentation timeslot according to current timeslot
                    next_timeslot_list = [primary + n + 60 * base + 15 * x for x in range(4)]
                    if any(time in timeslot and time not in checked for time in next_timeslot_list):  # if any next consecutive timeslot exist in staff's schedule
                        checked += next_timeslot_list   # record these all next possible timeslot as checked
                        if n > 3:  # if the consecutive has reach more than staff's preference in number of consecutive presentation
                            score += 10
                        if primary + n + 60 * base == 15:  # if the consecutive has reach end of day time period
                            break
                        else:
                            n += 1  # increment n to look for next consecutive presentation timeslot
                    else:  # if none next consecutive timeslot exist in staff's schedule then drop the consecutive count
                        break
    return score

def sc02(staff_timeslot):
    """check staff with number of days"""
    score = 0
    for _, timeslot in staff_timeslot.items():  # traverse and check each staff
        day_counter = {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0}   # to store each day's presentation
        for time in timeslot:   # traverse and check each timeslot
            if 0 < time < 61:   # increment counter by distribute timeslot into corresponding days
                day_counter["Monday"] += 1
            elif 60 < time < 121:
                day_counter["Tuesday"] += 1
            elif 120 < time < 181:
                day_counter["Wednesday"] += 1
            elif 180 < time < 241:
                day_counter["Thursday"] += 1
            else:
                day_counter["Friday"] += 1
        for _ in range(2):  # find the 2 days with lowest number of presentation to be used as penalty
            min_val = min(day_counter.values())
            min_day = min(day_counter.keys(), key=lambda x: day_counter[x])
            score += min_val * 10   # the lowest number of presentation will be used as penalty
            day_counter.pop(min_day)    # pop out the day with lowest number of presentation to find 1st and 2nd day lowest
    return score

def sc03(staff_timeslot):
    """check staff with change of venue"""
    score = 0
    for staff, timeslot in staff_timeslot.items():  # traverse and check each staff
        if sc03_csv[staff-1] == 'yes':  # if the staff preference not to change
            timeslot = [x % 60 for x in timeslot]   # %60 to eliminate days distribution
            venue_counter = {"VR": 0, "MR": 0, "IR": 0, "BJIM": 0}  # to store each venue's presentation
            for time in timeslot:  # traverse and check each timeslot
                if 1 <= time <= 15:  # increment counter by distribute timeslot into corresponding venue
                    venue_counter["VR"] += 1
                elif 16 <= time <= 30:
                    venue_counter["MR"] += 1
                elif 31 <= time <= 45:
                    venue_counter["IR"] += 1
                else:
                    venue_counter["BJIM"] += 1
            max_venue = max(venue_counter.keys(), key=lambda x: venue_counter[x])   # find the venue with most number of presentation taking place
            venue_counter.pop(max_venue)    # pop out venue with the most number of presentation taking place
            score += sum(venue_counter.values()) * 10    #all venues except venue with most number of presentation taking place will be used as penalty
    return score

def crossover(parent1_ind, parent2_ind, population):
    """crossover 2 parents chromosome function"""
    child1, child2 = [], []
    reservoir1, reservoir2 = [], [] # to store extra timeslot
    # insert parent1 and parent2 timeslot into child1 and child2 with probability
    for x in range(118):
        choice = random.choice((0, 1))  # probability to insert parent1 to child1 and parent2 to child2 or vice versa
        _, timeslot1 = population[parent1_ind][x]
        _, timeslot2 = population[parent2_ind][x]
        if choice:  # if probability is parent1 to child2 and parent2 to child1
            if timeslot2 not in child1: # if timeslot doesn't exist in child1 then insert the timeslot
                child1.append(timeslot2)
            else:   # if timeslot already exists in child1 then mark as 0 first
                child1.append(0)
                reservoir1.append(timeslot2)
            if timeslot1 not in child2: # if timeslot doesn't exist in child2 then insert the timeslot
                child2.append(timeslot1)
            else:   # if timeslot already exists in child2 then mark as 0 first
                child2.append(0)
                reservoir2.append(timeslot1)
        else:   # if probability is parent1 to child1 and parent2 to child2
            if timeslot1 not in child1: # if timeslot doesn't exist in child1 then insert the timeslot
                child1.append(timeslot1)
            else:   # if timeslot already exists in child1 then mark as 0 first
                child1.append(0)
                reservoir1.append(timeslot1)
            if timeslot2 not in child2: # if timeslot doesn't exist in child2 then insert the timeslot
                child2.append(timeslot2)
            else:   # if timeslot already exists in child2 then mark as 0 first
                child2.append(0)
                reservoir2.append(timeslot2)

    # a list of timeslot that is no repetition according to child1 and child2 timeslot
    non_existed = [x for x in range(1, 301) if x not in set(child1+child2)]
    # insert non-repetition timeslot to child1 and child2 with timeslot that is marked as 0
    for ind in range(118):
        if child1[ind] == 0:
            if len(reservoir2) != 0:
                child1[ind] = reservoir2[0]
                del reservoir2 [0]
            else:
                child1[ind] = random.choice(non_existed)
                non_existed.remove(child1[ind])
        if child2[ind] == 0:
            if len(reservoir1) != 0:
                child2[ind] = reservoir1[0]
                del reservoir1 [0]
            else:
                child2[ind] = random.choice(non_existed)
                non_existed.remove(child2[ind])

    # compile child1 and child2 into child_population and return it for further use
    result1, result2 = [], []
    for x in range(118):
        result1.append([population[parent1_ind][x][0], child1[x]])
        result2.append([population[parent1_ind][x][0], child2[x]])
    return result1, result2

def mutation(chromosome):
    """mutation 2 parents/childs chromosome function"""
    existed_timeslot = [x[1] for x in chromosome]   # list of current existed and scheduled timeslot in all presentation
    randomable_timeslot = [x for x in range(1, 301) if x not in (existed_timeslot or hc03_csv)] # list of non-existed and non-repetition timeslot
    index = random.randint(0, len(chromosome) - 1)  # randomly choose one chromosome out of population
    staff_timeslot = []
    for staff in chromosome[index][0].staff:
        staff_timeslot += hc04_csv[staff - 1]
    for timeslot in staff_timeslot:
        if timeslot in randomable_timeslot:
            randomable_timeslot.remove(timeslot)
    chromosome[index][1] = random.choice(randomable_timeslot)   # change the timeslot of a presentation group


# function to return key for any value
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

# Simple Searching Neighborhood
# It randomly changes timeslot of a class/lab
def ssn(solution):
    existed_timeslot = [x[1] for x in solution]  # list of current existed and scheduled timeslot in all presentation
    randomable_timeslot = [x for x in range(1, 301) if x not in (existed_timeslot or hc03_csv)]  # list of non-existed and non-repetition timeslot
    random.shuffle(randomable_timeslot)
    a = random.randint(0, len(solution) - 1)

    new_solution = copy.deepcopy(solution)
    new_solution[a][1] = randomable_timeslot.pop()
    if len(set([x[1] for x in new_solution])) < 118:
        conflict_repair(new_solution, randomable_timeslot)
    return [new_solution]


# Swapping Neighborhoods
# It randomy selects two classes and swap their time slots
def swn(solution):
    a = random.randint(0, len(solution) - 1)
    b = random.randint(0, len(solution) - 1)
    new_solution = copy.deepcopy(solution)
    temp = solution[a][1]
    new_solution[a][1] = solution[b][1]
    new_solution[b][1] = temp

    existed_timeslot = [x[1] for x in solution]  # list of current existed and scheduled timeslot in all presentation
    randomable_timeslot = [x for x in range(1, 301) if x not in (existed_timeslot or hc03_csv)]  # list of non-existed and non-repetition timeslot

    if len(set(existed_timeslot)) < 118:    # if a set contains duplicate values means that there are clashing slots
        conflict_repair(new_solution, randomable_timeslot)
    # print("Diff", solution)
    # print("Meiw", new_solution)
    return [new_solution]


# function to repair clashing slots
def conflict_repair(solution, randomable_timeslot):
    existed_timeslot = [x[1] for x in solution]  # booked slots for the presentations
    oc_set = set()  # set to keep the non clashing slots
    conflicted_slots_index = []  # index for clashing slots
    for idx, val in enumerate(existed_timeslot):  # loop to find all classhing slots index, non clashing slots and clashing slots
        if val not in oc_set:
            oc_set.add(val)
        else:
            conflicted_slots_index.append(idx)

    random.shuffle(randomable_timeslot)
    while len(set(existed_timeslot)) < 118:
        solution[conflicted_slots_index.pop()][1] = randomable_timeslot.pop()  # select one of the clashing slot and replace with the random slot
        existed_timeslot = [x[1] for x in solution]  # calculate the booking slots again to make sure no clash again


# function for acceptence probability
def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp((old_cost - new_cost) / temperature)


def simulated_annealing(target):
    alpha = 0.9
    T = 1.0

    solution = [target]  # as simulated annealing is a single-state method
    old_cost = evaluate(solution)   # cost of current solution
    for __n in range(10):
        new_solution = ssn(solution[0])  # generate the new solution by simple searching neighbourhood
        new_cost = evaluate(new_solution)
        ap = acceptance_probability(old_cost[0][1], new_cost[0][1], T)  # calculating acceptance probability
        if ap > random.random():
            solution = new_solution
            old_cost = new_cost
        T = T * alpha  # rising the temperature of the metal(solution)
    return solution[0]

def generate_result_csv(final_result):
    """generate result in result.csv function"""
    # format to be used in csv
    header_format = ["Day", "Venue", "0900-0930", "0930-1000", "1000-1030", "1030-1100", "1100-1130", "1130-1200", "1200-1230",
                    "1230-1300", "1400-1430", "1430-1500", "1500-1530", "1530-1600", "1600-1630", "1630-1700", "1700-1730"]
    day_format = ['Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']
    venue_format = ['VR', 'MR', 'IR', 'BJIM']
    # reverse sort the final result
    reverse_sorted_final_result = sorted(final_result, key=lambda tup: tup[1], reverse=True)
    csv_arrangement = []    # to store all result in csv format
    # placing result into csv data list
    for x in range(20):
        csv_arrangement.append([])
        if x % 4 == 0:
            csv_arrangement[x].append(day_format.pop())
            csv_arrangement[x].append(venue_format[x % 4])
        else:
            csv_arrangement[x].append(None)
            csv_arrangement[x].append(venue_format[x % 4])
        for y in range(15):
            if len(reverse_sorted_final_result) >= 1:   # if there is any more result to insert
                group, timeslot = reverse_sorted_final_result[-1]
                primary = (timeslot - 1) % 15
                base = (timeslot - 1) // 15
                if base == x and primary == y:      # if the presentation timeslot same as cell coordinate then insert presentation number
                    csv_arrangement[x].append('P'+str(group.name))
                    reverse_sorted_final_result.pop()
                else:   # if there is no presentation number
                    csv_arrangement[x].append(None)
            else:   # if there is no presentation number
                csv_arrangement[x].append(None)
    # writing data list into the csv
    try:
        with open("data/result.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header_format)
            writer.writerows(csv_arrangement)
        return True
    except PermissionError:
        return False

"""main body of program"""
clear = lambda : os.system('cls')
cmds = ["Command list: ",
    "              run             : Run program",
    "              result          : Display result from result.csv (latest run of program)",
    "              csv             : Open csv file",
    "              help            : Display command list",
    "              exit            : Exit program\n"]
print("Hi user, this is our CPT 244 Assignment 2: Presentation Scheduling Using Genetic Algorithm/ Hybrid System".center(120, '_'))
print("\n".join(cmds))
while True:
    execute = False
    cmd = input()
    clear()
    if cmd == 'run':    # run program
        generation_limit = input("Please input the number of generation limit (1-300): ")
        if generation_limit.isnumeric() and int(generation_limit) > 0:
            population_number = input("Please input the number of population (10-100): ")
            if population_number.isnumeric() and int(population_number) >= 10:
                execute = True
            else:   # invalid population number input
                print("Population number must be >= 10. Please try again.\n")
        else:   # invalid generation limit input
            print("Generation limit must be > 0. Please try again.\n")
    elif cmd == 'exit': # exit program
        print("\n\n\nThank you, goodbye! Have a nice day :D".center(60))
        time.sleep(2)
        exit()
    elif cmd == 'help':  # display command list
        print("\n".join(cmds))
    elif cmd == 'csv':  # open csv
        path = os.getcwd()
        str_path = ''
        for ind, ch in enumerate(path):
            if ord(ch) == 92:
                str_path += ch
            str_path += ch
        str_path += "\\data\\result.csv"
        if os.path.exists(str_path):
            os.system(str_path)
            print("\n".join(cmds))
        else:
            print('There is no previous run of program or no result.csv is found. Please make sure result.csv is in data folder.\n')
    elif cmd == 'result':  # display result from csv
        try:
            with open('data/result.csv', 'r') as file:  # retrieve file
                reader = csv.reader(file)
                result_csv = [row for row in reader]
            output = PrettyTable(allign='l')  # prettytable
            output.field_names = result_csv.pop(0)
            for row in result_csv:
                output.add_row(row)
            print(output)
        except FileNotFoundError:
            print('There is no previous run of program or no result.csv is found. Please make sure result.csv in is data folder.\n')
    else:   # invalid command input
        print("Invalid input, please try again. Type 'help' for all command list.\n")

    if execute: # run program if inputs are valid
        # mapping of codes to numbers for easy processing
        staff_code_to_num = {"S{:03d}".format(i + 1): i + 1 for i in range(47)}
        room_code_to_num = {"VR": 1, "MR": 2, "IR": 3, "BJIM": 4}
        presentation_code_to_num = {"P{}".format(i + 1): i + 1 for i in range(118)}

        # read SupExaAssign for presentation information
        with open('data/SupExaAssign.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)    # skip reading header of the csv
            SupExaAssign = [row for row in reader]
        # storing presentation information in data model
        staff_to_presentation = Presentation.presentation = [Presentation(presentation_code_to_num[j[0]], [staff_code_to_num[j[1]], staff_code_to_num[j[2]],
                                                                           staff_code_to_num[j[3]]]) for j in SupExaAssign]
        # read HC03.csv for venue unavailability
        hc03_csv = []
        with open('data/HC03.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                for x in set(row):
                    if str(x).isnumeric():
                        hc03_csv.append(int(x))
        # read HC04.csv for staff unavailability
        with open('data/HC04.csv', 'r') as file:
            reader = csv.reader(file)
            hc04_csv = [[int(x) for x in row if str(x).isnumeric()] for row in reader]
        # read SC03.csv for change of venue
        with open('data/SC03.csv', 'r') as file:
            reader = csv.reader(file)
            sc03_csv = [pref for _, pref in reader]

        print("\nExecuting Hybrid System (Genetic Algorithm & Simulated Annealing)...\n")
        start_time = time.time()
        final_result = genetic_algorithm(int(generation_limit), int(population_number)) # run genetic algorithm
        print("Time taken (s):", (time.time() - start_time))
        print("\nGenerating final result...")
        if generate_result_csv(final_result):   # generate final result
            print("\nResult is computed. Please type 'result' to display it here or 'csv' to open the csv!\n")
        else:
            print('\nGenerate result fail. Please close the result.csv first.\n')
