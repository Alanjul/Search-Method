
import numpy as np  #Helps in working with arrays and generate random numbers
import copy #to create copies
from typing import Dict, List, Tuple, Set, Optional #data structures for storing the data
import matplotlib.pyplot as plt

class Data:
    """Representing all the data for scheduling"""
    def __init__(self):
        self.rooms = [] #stores all rooms
        self.meeting_times = [] #store time
        self.facilitators = [] #store facilitators avaialable
        self.activities = [] #store the activities
    def initialization(self):
        """Initialization method used to initialize all the activites, time,
        facilitators and rooms"""
        times = ["10:00AM", "11:00AM", "12:00PM", "1:00PM", "2:00PM", "3:00PM"]
        #looping through available time
        for i, time in enumerate(times):
            self.meeting_times.append(MeetingTime(f"TM{i+1}", time))
        #initialize rooms with capacities
        self.rooms.append(Room("Slater", "003", 45))
        self.rooms.append(Room("Roman", "216", 30))
        self.rooms.append(Room("Loft", "206", 75))
        self.rooms.append(Room("Roman", "201", 50))
        self.rooms.append(Room("Loft", "310", 108))
        self.rooms.append(Room("Beach", "201", 60))
        self.rooms.append(Room("Beach", "301", 75))
        self.rooms.append(Room("Logos","301", 450))
        self.rooms.append(Room("Frank","119", 60))

        #initialize the facilitators
        self.facilitators.append(Facilitator("LO","Lock"))
        self.facilitators.append(Facilitator("GL", "Glen"))
        self.facilitators.append(Facilitator("BA", "Banks"))
        self.facilitators.append(Facilitator("RI", "Richards"))
        self.facilitators.append(Facilitator("SH", "Shaw"))
        self.facilitators.append(Facilitator("SI", "Singer"))
        self.facilitators.append(Facilitator("UT", "Uther"))
        self.facilitators.append(Facilitator("TY", "Dr.Tyler"))
        self.facilitators.append(Facilitator("NU", "Numen"))
        self.facilitators.append(Facilitator("ZI", "Zeldin"))

        #initializing activities with their preferred facilitators
        preferred_facilitators = [self.facilitators[1],self.facilitators[0],self.facilitators[2], self.facilitators[9]]
        other_facilitators = [self.facilitators[8], self.facilitators[3]]


        #SLA100A Activities
        self.activities.append(Activity("A1","SLA100", "A",50, preferred_facilitators,
                                       other_facilitators))
        #section 100B
        self.activities.append(Activity("B1", "SLA100",  "B",50, preferred_facilitators,
                                       other_facilitators))

        #activity SLA191A
        self.activities.append(Activity("A2", "SLA191" , "A",50, preferred_facilitators,
                                        other_facilitators))
        # activity SLA191B
        self.activities.append(Activity("B2", "SLA191" , "B",50, preferred_facilitators,
                                        other_facilitators))
        #activity for SLA201
        preferred_facilitators_201 = [self.facilitators[1], self.facilitators[2], self.facilitators[9],
                                      self.facilitators[4]]
        other_201 = [self.facilitators[8], self.facilitators[3], self.facilitators[5]]

        #appending activities of 201
        self.activities.append(Activity("B3","SLA201",None,
                                        50,preferred_facilitators_201, other_201))

        #activities for 291
        preferred_facilitators_291 = [self.facilitators[0], self.facilitators[2], self.facilitators[9],
                                      self.facilitators[5]]
        other_291 = [self.facilitators[8], self.facilitators[3], self.facilitators[4], self.facilitators[7]]

        #appending the facilitator , enrollment to section SLA291
        self.activities.append(Activity("B4","SLA291",None,50,
                                        preferred_facilitators_291,other_291))

        #activities in section 303
        section_303_facilitators =[self.facilitators[1], self.facilitators[9], self.facilitators[2]]
        facilitators = [self.facilitators[8], self.facilitators[5], self.facilitators[4]]

        #section 303 to activities
        self.activities.append(Activity("B5","SLA303", None,60, section_303_facilitators,
                                        facilitators))
        #available facilitators for section 304
        preferred_facilitators_304 = [self.facilitators[1],self.facilitators[2], self.facilitators[7]]
        facilitators_304 =  [self.facilitators[8], self.facilitators[5], self.facilitators[4], self.facilitators[3],
                            self.facilitators[6], self.facilitators[9]]

        #append section 304 to activities
        self.activities.append(Activity("B6", "SLA304", None,25, preferred_facilitators_304,
                                        facilitators_304 ))


        # available facilitators for section 394
        preferred_facilitators_394 = [self.facilitators[7], self.facilitators[5] ]
        facilitators_394 = [ self.facilitators[3], self.facilitators[9]]

        #append section 394 to activities
        self.activities.append(Activity("C1", "SLA394", None,24, preferred_facilitators_394,
                                        facilitators_394))

        #faciliators, enrollement for section 449
        preferred_facilitators_449 = [self.facilitators[7], self.facilitators[5], self.facilitators[4]]
        facilitators_449 = [self.facilitators[9], self.facilitators[6]]

        # append section 449 to activities
        self.activities.append(Activity("C2", "SLA449", None,60, preferred_facilitators_449,
                                        facilitators_449))

        #faciliators, enrollement for section 451
        facilitators_451 = [self.facilitators[9], self.facilitators[6], self.facilitators[3], self.facilitators[2]]

        #apend to activities for section 451
        self.activities.append(Activity("C3", "SLA451", None,100, preferred_facilitators_449,
                                        facilitators_451))


class MeetingTime:
    """For time slot for each activities"""
    def __init__(self,id:str, time:str):
        self.__time = time
        self.__id = id
    def getTime(self):
        return self.__time
    def __str__(self): #return string representation
        return self.__time

    def getId(self):
        return self.__id


class Room:
    """The room holds a room where the classes will be hold"""
    def __init__(self, building: str,  number:str,maximum: int, ):
        """Number represent the room number and maximum represent the capacity the room can hold
        building represent the name of the building"""
        self.__number = number
        self.__maximum = maximum
        self.__building = building
    def getNumber(self) -> str:
        #return the room number
        return self.__number
    def getMaximum(self) -> int:
        #return the maximum capacity the room can hold
        return self.__maximum
    def getBuilding(self) ->str:
        #return the building if present else return N/A
        #check if there any building listed
        return self.__building if self.__building else "N/A"

    def __str__(self):
        if self.__building:
            return f"{self.getBuilding()}{self.getNumber()}"
        return f"{self.getNumber()} "
    def __repr__(self):
        """Repr method for debugging"""
        return self.__str__()

#class facilitator for storing information about
# facilitators available to oversee a different activities
class Facilitator:
    """Hold the facilitors for each class"""
    def __init__(self,id,  name):
        self.__name = name #hold the name of the faciltator
        self.id = id # to hold id
    def getName(self) -> str:
        return self.__name
    def getId(self) :
        return self.id
    def __str__(self) -> str:
        """return string representation of facilitators names"""
        return f"Name: {self.getName()}"
    def __repr__(self):
        return self.__str__()


class Activity:
    """Representating activity to be scheduled"""
    def __init__(self, id:str, name:str, section, expected_enrollment: int,
                 preferred_facilitor: List[Facilitator],
                 other_facililator : List[Facilitator], ):
        #initialization of the private methods
        self.__id = id
        self.__name  = name
        self.section = section
        self.__expected_enrollment=  expected_enrollment
        self.__preferred_facilitator = preferred_facilitor
        self.__other_facilitator = other_facililator


    #get methods to return the id , name, enrollment, facilitators and section
    def getId(self) -> str:
        return self.__id
    def getName(self) -> str:
        return self.__name
    def getExpectedEnrollment(self) -> int:
        return self.__expected_enrollment
    def getPreferredFacilitator(self) -> List[Facilitator]:
        return  self.__preferred_facilitator
    def getOtherFacilitator(self) -> List[Facilitator]:
        return self.__other_facilitator

    #string method to return string representation of the name and the section
    def __str__(self)->str:
        if self.section is not None:
            return f"Name : {self.getName()} section: {self.section}"
        return f"Name: {self.getName()}"
    def __repr__(self):
        return self.__str__()

class ActivitySlot:
    """Activity slot class representing scheduled activities with asssigned room, time and facilitator"""
    def __init__(self, activity: Activity, room: Room=None, meeting_time: MeetingTime=None,
                 facilitator:Facilitator=None):
        self.__activity = activity
        self.__room = room
        self.__meeting_time = meeting_time
        self.__facilitator = facilitator
    #setter methods
    def setActivity(self, activity: Activity):
        self.__activity = activity
    def setRoom(self, room: Room):
        self.__room = room

    def setMeetingTime(self, meeting_time: MeetingTime):
        self.__meeting_time = meeting_time
    def setFacilitator(self, facilitator: Facilitator):
        self.__facilitator = facilitator

     #getter methods
    def getActivity(self) -> Activity:
        return self.__activity
    def getRoom(self) -> Room:
        return self.__room
    def getMeetingTime(self) -> MeetingTime:
        return self.__meeting_time
    def getFacilitator(self) -> Facilitator:
        return self.__facilitator
    def __str__(self) -> str:
        activity_str = str(self.getActivity())
        room_str = str(self.getRoom()) if self.getRoom() else "Unassigned Room"
        time_str = str(self.getMeetingTime()) if self.getMeetingTime() else "Unscheduled Time"
        facilitator = self.getFacilitator()
        if not facilitator:
            facilitator_str = "No Facilitator Assigned"
        else:
            facilitator_str = str(facilitator)

        return f"{activity_str:<40} | {room_str:<25} | {time_str:<15} | {facilitator_str:<15}"


    def __repr__(self):
        return self.__str__()




class Schedule:
    """Scheduling activites one per slot"""
    def __init__(self, activity_slots: List[ActivitySlot]=None):
        self.__activity_slots = activity_slots if activity_slots else []  #assign empty is there's no slot
        self.fitness = 0.0
        self.normalized_fitness = 0.0

    def setActivity_slot(self, activity_slots:List[ActivitySlot]):
        self.__activity_slots = activity_slots if activity_slots else [] #assign empty slot

    #get the activity
    def getSlot(self, activity: Activity) -> ActivitySlot | None: #return specific slots
        for slot in self.__activity_slots:
            if slot.getActivity() == activity:
                return slot
        return None

    @property
    def activity_slots(self): #returns all available slots
        return self.__activity_slots

    def addSlot(self, slot: ActivitySlot): #to add a single slot for the activity
        self.__activity_slots.append(slot)
    def  __str__(self):
        result = "Schedule (Fitnes: {:6f}):\n".format(self.fitness)

        #sorting time and room numbers
        slot_sorted = sorted(self.__activity_slots, key=lambda x: (x.getMeetingTime().getTime()
                                                                   if x.getMeetingTime() else "",
                             x.getRoom().getNumber() if x.getRoom() else ""
                             ))
        header= f"{'Course':<10} | {'Section':<7} | {'Room':<30} | {'Cap.':<5} | {'Time':<10} | {'Facilitator':<25}\n"
        result += header
        result += "-" * len(header) + "\n"

        for slot in slot_sorted:
            facilitator = slot.getFacilitator()


            activity = slot.getActivity()
            section = activity.section if activity.section is not None else "N/A"
            course_name = f"{activity.getName():<10}"

            room = slot.getRoom()
            if facilitator:
                facilitator_name = f"{facilitator.getName():<25}"
            else:
                facilitator_name = "No Facilitator Assigned"
            if room:
                room_str = f"{room.getBuilding()} {room.getNumber():<4}"
                capacity = f"{room.getMaximum():<5}"
            else:
                room_str = "Unassigned Room"
                capacity = "N/A"

            meeting_time = slot.getMeetingTime()
            time = str(meeting_time) if meeting_time else "Unscheduled"



            result += f"{course_name} | {section:<7} | {room_str:<30} | {capacity} | {time:<10} | {facilitator_name}\n"

        return result


#GeneticAlgorithm class designed to generate schedules using genetic algorithms
class GeneticAlgorithm:
    """The genetic algorith to do the scheduling"""
    def __init__(self,data, population_size:int=500, mutation_rate: float= 0.01, priority_size:int= 10):
        self.data = data #to hold information needed for change
        self.population_size = population_size #holds schedules the algorithm will use
        self.mutation_rate = mutation_rate #To introduce variation
        self.priority_size = priority_size #Priority schedules that will remain unchanged



    def population(self) -> List[Schedule]: #created the first generation
        #create an empty dictionary to hold the  population
        pop = []
        for _ in range(self.population_size):
            schedule = Schedule() #instantiating the schedule class

            for activity in self.data.activities:
                slot = ActivitySlot(activity)

                # Randomly assign a room
                available_rooms = [room for room in self.data.rooms if
                                   room.getMaximum() >= activity.getExpectedEnrollment()]
                if not available_rooms:
                    available_rooms = self.data.rooms #use any rooms
                slot.setRoom(np.random.choice(available_rooms))

                # Randomly assign a meeting time
                slot.setMeetingTime(np.random.choice(self.data.meeting_times))

                # Weighted random selection: prefer preferred_facilitators
                preferred = activity.getPreferredFacilitator()
                others = activity.getOtherFacilitator()


                if preferred or others:
                    facilitators_pool = preferred + others
                    weights = [0.7] * len(preferred) + [0.3] * len(others)
                    total = sum(weights)
                    normalized_weights = [w / total for w in weights]  # normalize so they sum to 1
                    facilitator = np.random.choice(facilitators_pool, p=normalized_weights)
                    slot.setFacilitator(facilitator)
                else:
                    # Fallback: random facilitator from all if none are listed
                    slot.setFacilitator(np.random.choice(self.data.facilitators))

                # Add the slot to the schedule
                schedule.addSlot(slot)

            # Score the schedule
            self.calculate_fitnes(schedule)

            # Add to population
            pop.append(schedule)
        return pop  #return the entire population
    def calculate_fitnes(self, schedule:Schedule):
        """Calculate class  calculates the fitness level"""
        fitness = 0.0  # initialize the fitness start

        #track room conflicts
        room_pairs = {}

        #Track facilitators schedules
        facilitator_schedules = {}

        #find sla1 and 191 sections
        sla101_slots = []
        sla191_slots = []

        #processing each activity in the slot
        for slot in schedule.activity_slots:
            activity = slot.getActivity()
            room = slot.getRoom()
            time = slot.getMeetingTime()
            facilitator = slot.getFacilitator()

            #Track SLA slots
            if activity.getName() =="SLA100":
                sla101_slots.append(slot)
            elif activity.getName() == "SLA191":
                sla191_slots.append(slot)
            slot_fitness = 0.0

            #checking room and time conflicts
            if time and room:
                key = (f"{room.getNumber()} -{time.getTime()}")
                if key in room_pairs:
                    slot_fitness -= 0.5 # conflicting with another activity
                else:
                    room_pairs[key] = activity
            #checking for size
            if room and activity:
                #checking for capacity
                if room.getMaximum() < activity.getExpectedEnrollment():
                    slot_fitness -=0.5 #room too small
                elif room.getMaximum() > 6* activity.getExpectedEnrollment():
                    slot_fitness -=0.4#room is too big
                elif room.getMaximum() > 3 * activity.getExpectedEnrollment():
                    slot_fitness -= 0.2 # room too big
                else:
                    slot_fitness +=0.3 #apppropriate size

            # Facilitator-related checks
            if facilitator and time:
                fid = facilitator.getId()

                # Initialize tracking for this facilitator
                if fid not in facilitator_schedules:
                    facilitator_schedules[fid] = []

                    # Append time and activity
                facilitator_schedules[fid].append((time, activity))

                    #check for preferred facilitator
                if facilitator  in activity.getPreferredFacilitator():
                    slot_fitness += 0.5 #preferred facilitator
                elif facilitator in activity.getOtherFacilitator():
                    slot_fitness +=0.2 #other listed facilitators
                else:
                    slot_fitness -= 0.1 #any other facilitator
            fitness += slot_fitness #get the total fitness

        #Facilitator load times
        for fid, schedule_items in facilitator_schedules.items():
            # Check for multiple activities at the same time
            time_dict = {}
            # Use facilitator_counts which contains (time, activity) tuples
            for time_obj, activity_obj in schedule_items:
                time_id = time_obj.getId()  # Extract time id
                if time_id not in time_dict:
                    time_dict[time_id] = []
                time_dict[time_id].append(activity_obj)

             #checking for facilitators activities per time slot
            for time_id, activities in time_dict.items():
                if len(activities) == 1:
                    fitness += 0.2
                else:
                    fitness -= 0.2

            #check for facilitator load
            if  fid == "TY" and len(schedule_items) < 2: #Exception for Dr. Tyler
                pass  #No penalty should be given
            elif len(schedule_items) <= 2:
                fitness -= 0.4
            elif len(schedule_items)> 4:
                fitness -= 0.5

            # Find consecutive time slots for facilitator
            # Sort activities by time
            sorted_items = sorted(schedule_items, key=lambda x: self.time_index(x[0]))

            # Look for consecutive slots
            for i in range(len(sorted_items) - 1):
                time1 = sorted_items[i][0]
                time2 = sorted_items[i + 1][0]

                # If times are consecutive (1 hour apart)
                if abs(self.time_index(time2) - self.time_index(time1)) == 1:
                    # Check for buildings - can the facilitator travel between them?
                    activity1 = sorted_items[i][1]
                    activity2 = sorted_items[i + 1][1]

                    slot1 = schedule.getSlot(activity1)
                    slot2 = schedule.getSlot(activity2)

                    if slot1 and slot2 and slot1.getRoom() and slot2.getRoom():
                        building1 = slot1.getRoom().getBuilding()
                        building2 = slot2.getRoom().getBuilding()

                        # If buildings require significant travel time
                        if ((building1 in ["Roman", "Beach"] and building2 not in ["Roman", "Beach"]) or
                                (building2 in ["Roman", "Beach"] and building1 not in ["Roman", "Beach"])):
                            fitness -= 0.4  # Penalty for difficult travel
        #special rules
        if len(sla101_slots) == 2:  #slots for sla101
            slot_a, slot_b = sla101_slots #unpack the two slots into a and b
            if slot_a.getMeetingTime() and slot_b.getMeetingTime():
                time_difference = self.calculate_time_difference(slot_a.getMeetingTime(),
                                                                 slot_b.getMeetingTime())
                #same time slot
                if time_difference == 0: #no time difference
                    fitness -= 0.5
                elif time_difference > 4:
                    fitness += 0.5

        #special rules for SLA191
        if len(sla191_slots) == 2:  # slots for SLA 191
            slot_a, slot_b = sla191_slots  # unpack the two slots into a and b
            if slot_a.getMeetingTime() and slot_b.getMeetingTime():
                time_difference = self.calculate_time_difference(slot_a.getMeetingTime(),
                                                                 slot_b.getMeetingTime())
                # same time slot
                if time_difference == 0:
                    fitness -= 0.5
                elif time_difference > 4:
                    fitness += 0.5

        #special rule for SLA191 and 101
        for slot101 in sla101_slots:
            for slot191 in sla191_slots:
                if slot101.getMeetingTime() and slot191.getMeetingTime():
                    time_difference = self.calculate_time_difference(slot101.getMeetingTime(),
                                                                 slot191.getMeetingTime())
                    if time_difference == 0:
                        fitness -= 0.25 #same time slots
                    elif time_difference == 1:
                        fitness += 0.5 #consecutive time slots

                    #checking for consecutive time slots
                    if time_difference == 1:
                        room1 = slot101.getRoom()
                        room2 = slot191.getRoom()
                        if room1 and room2:
                            building1 = room1.getBuilding()
                            building2 = room2.getBuilding()

                            #checking if the activities are in Roman or Beach
                            if ((building1 in ["Roman", "Beach"] and building2 not in ["Roman", "Beach"]) or
                                (building2  in ["Roman", "Beach"] and building1  not in ["Roman", "Beach"])):
                                fitness -= 0.4 #building too far apart for consecutive sessions

                    elif time_difference == 2: #separated by 1 hour
                        fitness += 0.25
        #store the fitness in the schedule
        schedule.fitness = fitness
        return fitness




    def find_consecutive_times(self, times_list: List[Tuple[MeetingTime, Activity]])->List[Tuple[MeetingTime,
    MeetingTime]]:
        """The method is use to find back to back activities """
        if not times_list:
            return []

            # Sort times
        times_list.sort(key=lambda x: self.time_index(x[0]))
        consecutive = []

        #finding consecutive time slots:
        for i in range( len(times_list)-1):
            time1 = times_list[i][0]
            time2 = times_list[i+1][0]
            if (self.time_index(time2) - self.time_index(time1)) == 1:
                consecutive.append((time1, time2))

        return consecutive
    def  calculate_time_difference(self, time1:MeetingTime, time2: MeetingTime) -> int:
        """Calculating the time difference"""
        id1 = self.time_index(time1)
        id2 = self.time_index(time2)

        return abs(id1-id2) #return the time difference


    def time_index(self, time: MeetingTime) -> int :
        """This method returns time converted in numerical form"""
        for i, value in enumerate(self.data.meeting_times):
            if value.getId() == time.getId():
                return i
        raise ValueError(f"MeetingTime with ID {time.getId()} not found in master list.")
    def softmax_activation_function(self, population:List[Schedule]) -> List[Schedule]:
        """the function applies softmax normalization to convert fitness to probabilities"""
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True) #sort the population in descending order

        #getting fitness value
        fitness_value = np.array([s.fitness for s in sorted_population])
        #applying normalizarion
        scaled_fitness = fitness_value * 10 #scale
        exp_fitness = np.exp(scaled_fitness - np.max(scaled_fitness)) #substract the max
        softmax_probs = exp_fitness /np.sum(exp_fitness)

        #select individuals based on the probabilities
        indices = np.random.choice(len(sorted_population), size=len(sorted_population), p=softmax_probs)
        selected = [sorted_population[i] for i in indices]
        return selected
    def cross_over(self, parent1: Schedule, parent2: Schedule) -> Schedule:
<<<<<<< HEAD
        #performing a cross over to combine two parents  to create two ospring parts of their genetic material
        child = Schedule() #initializing the schedule clas
=======
        #This function performs a cross over to combine two parents  to create two offspring parts of their genetic material
        child = Schedule() #initializing the schedule class
>>>>>>> 18a176fd4f4e6c79998ce95b5ad6239d89b07bbf

        #for each activity , randomly choice a either parent
        for activity in self.data.activities:
            parent1_slot = parent1.getSlot(activity)
            parent2_slot = parent2.getSlot(activity)

            #create a new slot
            if parent1_slot is None or parent2_slot is  None:
                child_slot =ActivitySlot(activity)
                suitable_rooms = [room for room in self.data.rooms if
                                  room.getMaximum()>= activity.getExpectedEnrollment()]
                if suitable_rooms:
                    child_slot.setRoom(np.random.choice(suitable_rooms))
                else:
                    child_slot.setRoom(np.random.choice(self.data.rooms))

                child_slot.setMeetingTime(np.random.choice(self.data.meeting_times))

                available = activity.getPreferredFacilitator() + activity.getOtherFacilitator()
                if not available:
                    available = self.data.facilitators
                    child_slot.setFacilitator(np.random.choice(available))
            else:
                child_slot = ActivitySlot(activity)
                #inheriting attributes from the parents
                if np.random.random() < 0.5: #50% of inheriting from the parents
                    child_slot.setRoom( parent1_slot.getRoom())
                else:
                    child_slot.setRoom( parent2_slot.getRoom())

                if np.random.random() < 0.5:
                    child_slot.setMeetingTime(parent1_slot.getMeetingTime())
                else:
                    child_slot.setMeetingTime(parent2_slot.getMeetingTime())

                if np.random.random() < 0.5:
                    child_slot.setFacilitator(parent1_slot.getFacilitator())
                else:
                    child_slot.setFacilitator(parent2_slot.getFacilitator())

            #adding slot to child schedule
            child.addSlot(child_slot)
        return child


    def mutation(self,schedule:Schedule):
        # The functions applies random mutations to the schedule
        for slot in schedule.activity_slots:
            activity = slot.getActivity()
            # Room mutation
            if np.random.random() < self.mutation_rate:
                slot.setRoom(np.random.choice(self.data.rooms))

            # Time mutation
            if np.random.random() < self.mutation_rate:
                slot.setMeetingTime(np.random.choice(self.data.meeting_times))

            # Facilitator mutation (only from preferred + other pools)
            if np.random.random() < self.mutation_rate:
                slot.setFacilitator(np.random.choice(self.data.facilitators))

        return schedule
    def evolve(self, population: List[Schedule]) -> List[Schedule]:
        #Ensure all schedules have updated fitness
        for schedule in population:
            self.calculate_fitnes(schedule)

        #sort in descending order
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)

        #keeping on elite individuals
        elite_count = self.priority_size
<<<<<<< HEAD
        new_population = sorted_population[:elite_count].copy() #keeping individuals with highest factors

        #using the softmax to select the parents
        selection_pool = self.softmax_activation_function(sorted_population)

=======
        new_population = selected_population[:elite_count] #keeping individuals with highest factors
>>>>>>> 18a176fd4f4e6c79998ce95b5ad6239d89b07bbf
        #Generate new schedule
        while len(new_population) < self.population_size:
            # Select two distinct parents
            parents= np.random.choice(len(selection_pool), 2, replace=False)
            parent1, parent2 = selection_pool[parents[0]], selection_pool[parents[1]]

            #create a child through cross
            child = self.cross_over(parent1, parent2)

            #apply mutation to child
            child = self. mutation(child)

            #calculate the fitness of the new child
            self.calculate_fitnes(child)

            #add it to the new population
            new_population.append(child)

        return new_population


#main program
def main():
    #creating data instance and initializing it
    data = Data()
    data.initialization()
    #initial population
    population_size = 500
    mutation_rate = 0.01
    elite_size = 10
    max_generations = 100
    MIN_MUTATION_RATE = 0.0001  # set a floor for mutation rate

    algorithm = GeneticAlgorithm(data,population_size, mutation_rate, elite_size)

    #initialization of the population
    print("Initializing the population")
    population = algorithm.population()

    # Track best and average fitness
    best_fitness_history = []
    avg_fitness_history = []

    # Get initial statistics
    best_schedule = max(population, key=lambda x: x.fitness)
    best_fitness = best_schedule.fitness
    avg_fitness = sum(s.fitness for s in population) / len(population)

    print(f"Generation 0: Best Fitness = {best_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")
    best_fitness_history.append(best_fitness)
    avg_fitness_history.append(avg_fitness)

    #Initializing the generation counter
    generation = 1 # tracking how many  iterations
    threshold_improvement = 0.01 # improvement threshold
    converged = False  #track if the algorithm converged
    previous_best_fitness = best_fitness
    while generation <  max_generations and not converged:
        #population  evolving
        population = algorithm.evolve(population)

        #calculate statistics
        best_schedule = max(population, key=lambda x : x.fitness) #individual with highest fitness
        best_fitness = best_schedule.fitness

        average_fitness = sum(s.fitness for s in population) / len(population) #average  cross the entire generation

        print(f"Generation {generation}: Best Fitness = {best_fitness:.5f}, average Fitnes = {average_fitness:.5f}")
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(average_fitness)

        #check for 100 generation
        if generation >= max_generations: #checking for convergence
            #calculate the improvement percentage
            # by substracting the new value from old valued  divided by the old value
            improvement =(avg_fitness_history[-1] - avg_fitness_history[-100]) /abs(avg_fitness_history[-100])
            print(f"Improvement since generation {generation - 100}: {improvement:.4%}") #after hundred generation

            if improvement <  threshold_improvement:
                print(f"Converged with less than { threshold_improvement:.1%} improvement")
                converged=True

        if best_fitness > previous_best_fitness:
                new_rate = algorithm.mutation_rate / 2
                algorithm.mutation_rate = max(new_rate, MIN_MUTATION_RATE)
                print(f"Reduced mutation to {algorithm.mutation_rate}")
        previous_best_fitness = best_fitness

        generation += 1


    #Generate the best schedule from the final generation
    best_schedule = max(population, key = lambda x: x.fitness)

    print("\nBest Schedule found")
    print(best_schedule)
    print(f"Fitness: {best_schedule.fitness:.3f}")
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, label="Best Fitness", color="green", linewidth=2)
    plt.plot(avg_fitness_history, label="Average Fitness", linestyle="--", color="blue")
    plt.title("Fitness Evolution Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    #display the statistics for the best schedule
    with open ("Best Schedule.txt", "w") as file:
        file.write(str(best_schedule))
        file.write(f"\nFitness: {best_schedule.fitness:.4f}")

    #Display the statistics for the best schedule
    print(f"\nDetailed Schedule Analysis:")
    facilitators_work = {}
    room_usage = {}
    times_slot = {}
    for slot in best_schedule.activity_slots:
        #count the facilitator load
        if slot.getFacilitator():
            facilitator_name = slot.getFacilitator().getName()
            if facilitator_name not in facilitators_work:
                facilitators_work[facilitator_name] = 0
            facilitators_work[facilitator_name] += 1

        #count room usage
        if slot.getRoom():
            room_num = slot.getRoom().getNumber()
            if room_num not in  room_usage:
                room_usage[room_num] =0
            room_usage[room_num] += 1

            #counting time slots
        if slot.getMeetingTime():
            time_slot = slot.getMeetingTime().getTime()
            if time_slot not in times_slot:
                times_slot[ time_slot] = 0
            times_slot[time_slot] += 1

    #Displaying Facilitators loads
    for  name, count in facilitators_work.items():
        print(f"{name}: {count} activities")

    #displaying Room usage
    for room, count in room_usage.items():
        print(f"Room{room}: {count} activities")

    #display time usage
    for time,count in times_slot.items():
        print(f"{time}: {count} activities")


if __name__=="__main__":
    main()
