#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 23:05:04 2021

@author: taraskulyavets
"""
import argparse
import numpy as np
import gym
from random import choice
from time import sleep
from copy import deepcopy


class NeuralLayer:

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        
    def get_response(self, x):
        u = np.matmul(self.weight, x)
        u_prim = u + self.bias
        y = 1 / (1 + np.exp(-u_prim))
        return y
    
    def mutate(self, factor):
        weight_mutation = (np.random.rand(*(self.weight.shape)) * 2 - 1) * factor
        bias_mutation = (np.random.rand(*(self.bias.shape)) * 2 - 1) * factor
        self.weight += weight_mutation
        self.bias += bias_mutation
        
    @staticmethod
    def generate_random_layer(input_size, layer_size):
        weight = np.random.rand(layer_size, input_size) * 2 - 1
        bias = np.random.rand(layer_size) * (-1)
        return NeuralLayer(weight, bias)
    
class Speciman:
    
    def __init__(self, layers):
        self.layers = layers
        self.score = 0
        
    def get_response(self, x):
        h = x
        for layer in self.layers:
            h = layer.get_response(h)
        return np.argmax(h)
    
    def mutate(self, factor):
        for layer in self.layers:
            layer.mutate(factor)
            
    def __lt__(self, other):
        return self.score < other.score
        
    
    @staticmethod
    def generate_random_speciman(input_size, layer_sizes):
        layers = []
        for layer_size in layer_sizes:
            layers.append(NeuralLayer.generate_random_layer(input_size, layer_size))
            input_size = layer_size
        return Speciman(layers)
    
def test_gym(simulation_delay):
    environment = gym.make("CartPole-v1")
    observation = environment.reset()
    
    total_reward = 0
    
    print(f"environment.observation_space {environment.observation_space}")
    print(f"environment.action_space {environment.action_space}")
    
    finished = False
    environment.render()
    while not finished:
        action = choice([0,1])
        observation, reward, finished, _ = environment.step(action)
        total_reward += reward
        print(f"{observation} -> {action}")
        sleep(simulation_delay)
        environment.render()
        
    print(f"Total reward is {total_reward}")
        
    
def test_speciman():
    layer = NeuralLayer.generate_random_layer(3, 5)
    print(layer.weight)
    print(layer.bias)
    test_input = np.array([0.3, 0.5, 1.0])
    print(f"Response is {layer.get_response(test_input)}")
    layer.mutate(0.5)
    
    speciman = Speciman.generate_random_speciman(3, [5,6,3,2])
    print(f"Response of multi layer network is {speciman.get_response(test_input)}")
    speciman.mutate(0.3)
    
def run_simulation(speciman, environment, assign_score = True, show = False, delay = 0.2):
    observation = environment.reset()
    total_reward = 0
    finished = False
    if show:
        environment.render()
    while not finished:
        action = speciman.get_response(observation)
        observation, reward, finished, _ = environment.step(action)
        total_reward += reward
        if show:
            environment.render()
            sleep(delay)
            
    if assign_score:
        speciman.score = total_reward
       
       
def assign_scores(population, environment, show_best = True):
    for speciman in population:
        run_simulation(speciman, environment)
    population.sort()
    if show_best:
        print(f"Best score is {population[-1].score}")
        run_simulation(population[-1], environment, show = True)
       
    
def selection(population, factor, elite = 1):
    
    breeders = []
    population.sort()
    
    desired_breeders_count = int(len(population) * factor)
    for _ in range(elite):
        breeders.append(population.pop(-1))
        
    while len(breeders) < desired_breeders_count:
        speciman_a = choice(population)
        speciman_b = choice(population)
        if speciman_a < speciman_b:
            breeders.append(speciman_b)
        else:
            breeders.append(speciman_a)
        try:
            population.remove(speciman_a)
            population.remove(speciman_b)
        except ValueError:
            pass # If spaciman a is same as b second remove will raise this exception
            
    return breeders
    
    
def reproduction(breeding_base, population_size, elite = 1, mutation_factor = 0.5):
    population = []
    breeding_base.sort()

    for i in range(elite):
        population.append(breeding_base[0 - (1 + i)])

    while len(population) < population_size:
        new_speciman = deepcopy(choice(breeding_base))
        new_speciman.mutate(mutation_factor)
        population.append(new_speciman)
        
    return population


def create_initial_population(population_size, input_size, output_size, hidden_layers_sizes):
    population = []
    hidden_layers_sizes.append(output_size)
    for _ in range(population_size):
        population.append(Speciman.generate_random_speciman(input_size, hidden_layers_sizes))
    return population


def run_neuroevolution(population_size, generations, elite_size, selection_factor, mutation_factor, show_best=True):
    
    environment = gym.make("CartPole-v1")
    observation = environment.reset()
    
    input_size = observation.shape[0]
    output_size = 2 #environment.action_space
    
    population = create_initial_population(population_size, input_size, output_size, [7])
    
    for generation in range(generations):
        assign_scores(population, environment)
        breeding_base = selection(population, selection_factor, elite_size)
        population = reproduction(breeding_base, population_size, elite_size, mutation_factor)
    
    
    
    
    
    

def main(args):
    if args.test_gym:
        test_gym(args.simulation_step_delay)
    elif args.test_speciman:
        test_speciman()
    else:
        run_neuroevolution(200, 20, 1, 0.5, 0.3)
    
def parse_arguments():
    parser = argparse.ArgumentParser(description=('Example of neuroevolution'))
    parser.add_argument('--test_gym',
            action='store_true',
            help='If set then only no neuroevolution will be run, only gum ai env tested.')
    parser.add_argument('--test_speciman',
            action='store_true',
            help='If set then only no neuroevolution will be run, only speciman implementation test.')
    parser.add_argument('--simulation_step_delay', type=float, default = 0.1,
                        help='Time in seconds between steps of simulation')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())

