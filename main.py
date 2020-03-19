from utils import *
from constants import *
import pygame
from itertools import cycle
import numpy as np
import random

from neuralnetwork import *
from genetics import *

currentPool = []
fitness = []
generation = 0

nextPipeX = -1
nextPipeHoleY = -1


def predictAction(height, dist, pipe_height, model_num):
    global currentPool
    # The height, dist and pipe_height must be between 0 to 1 (Scaled by SCREENHEIGHT)
    height = min(SCREENHEIGHT, height) / SCREENHEIGHT - 0.5
    dist = dist / 450 - 0.5  # Max pipe distance from player will be 450
    pipe_height = min(SCREENHEIGHT, pipe_height) / SCREENHEIGHT - 0.5
    neural_input = np.asarray([height, dist, pipe_height])
    neural_input = np.atleast_2d(neural_input)
    output_prob = currentPool[model_num].predict(neural_input, 1)[0]
    if output_prob[0] <= 0.5:
        # Perform the jump action
        return 1
    return 2


# Initialize all the models
currentPool = createModel(totalPlayers)
for idx in range(totalPlayers):
    fitness.append(-100)

Images = {}


def main():
    global Screen, FPSCLOCK, fitness, HITMASKS

    HITMASKS = {}

    pygame.init()
    pygame.display.set_caption('Flappy Bird')
    Screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    FPSCLOCK = pygame.time.Clock()

    Images['background'] = pygame.image.load(
        BACKGROUND_IMAGE[0]).convert_alpha()

    Images['base'] = pygame.image.load(BASE_IMAGE).convert_alpha()

    while True:

        Images['player'] = (
            pygame.image.load(PLAYER_IMAGE[0]).convert_alpha(),
            pygame.image.load(PLAYER_IMAGE[1]).convert_alpha(),
            pygame.image.load(PLAYER_IMAGE[2]).convert_alpha(),
        )
        lowerPipeImage = pygame.image.load(PIPE_IMAGE[0]).convert_alpha()
        upperPipeImage = pygame.transform.rotate(lowerPipeImage, 180)

        Images['pipes'] = (
            lowerPipeImage, upperPipeImage
        )

        HITMASKS['pipe'] = (
            getHitmask(Images['pipes'][0]),
            getHitmask(Images['pipes'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(Images['player'][0]),
            getHitmask(Images['player'][1]),
            getHitmask(Images['player'][2]),
        )

        movementInfo = {'baseX': 0,
                        'playerIndexGen': cycle([0, 1, 2, 1]),
                        'playerY': int((SCREENHEIGHT - Images['player'][0].get_height())/2)
                        }

        for idx in range(totalPlayers):
            fitness[idx] = 0
        info = maingame(movementInfo)
        nextGeneration(info)


def maingame(movementInfo):

    global fitness, nextPipeX, nextPipeHoleY

    running = True
    playerIndexGen = movementInfo['playerIndexGen']
    playerIndex = loopIter = 0
    alivePlayers = totalPlayers
    playersXList = []
    playersYList = []

    for idx in range(totalPlayers):
        playersXList.append(int(SCREENWIDTH*0.2))
        playersYList.append(movementInfo['playerY'])

    baseX = movementInfo['baseX']
    baseShift = Images['base'].get_width() - Images['background'].get_width()
    pipeX = -4
    pipes = getRandomPipe(Images)

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': pipes[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': pipes[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': pipes[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': pipes[1]['y']},
    ]

    nextPipeX = lowerPipes[0]['x']
    nextPipeHoleY = (
        lowerPipes[0]['y'] + (upperPipes[0]['y'] + Images['pipes'][0].get_height()))/2

    playerMaxVel = 8  # Max Descend Speed
    playersVel = []
    playersAcc = []  # Player's Downward Acceleration
    playerFlapAcc = -9
    playersFlapped = []
    playersState = []

    for idx in range(totalPlayers):
        playersVel.append(-9)
        playersFlapped.append(False)
        playersState.append(True)
        playersAcc.append(1)

    while True:
        Screen.blit(Images['background'], (0, 0))

        for idx in range(totalPlayers):
            if playersState[idx] == True and playersYList[idx] < 0:
                alivePlayers -= 1
                playersState[idx] = False

        if alivePlayers == 0:
            return {
                'baseX': baseX
            }

        for idx in range(totalPlayers):
            if playersState[idx]:
                fitness[idx] += 1

        nextPipeX += pipeX

        for idx in range(totalPlayers):
            if playersState[idx]:
                # print(playersYList[idx], nextPipeX, nextPipeHoleY, idx)
                if predictAction(playersYList[idx], nextPipeX, nextPipeHoleY, idx) == 1:
                    if playersYList[idx] > -2 * Images['player'][0].get_height():
                        playersVel[idx] = playerFlapAcc
                        playersFlapped[idx] = True

        crashTest = checkCrashed({'x': playersXList, 'y': playersYList, 'index': playerIndex},
                                 upperPipes, lowerPipes, Images, HITMASKS)

        for idx in range(totalPlayers):
            if playersState[idx] == True and crashTest[idx] == True:
                alivePlayers -= 1
                playersState[idx] = False

        if alivePlayers == 0:
            return {
                'baseX': baseX
            }

        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30

        for idx in range(totalPlayers):
            if playersState[idx] == True:
                playerMidPos = playersXList[idx]
                pipe_idx = 0
                for pipe in upperPipes:
                    pipeMidPos = pipe['x'] + Images['pipes'][0].get_width()
                    if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                        nextPipeX = lowerPipes[pipe_idx+1]['x']
                        nextPipeHoleY = (lowerPipes[pipe_idx+1]['y'] + (
                            upperPipes[pipe_idx+1]['y'] + Images['pipes'][pipe_idx+1].get_height())) / 2
                        fitness[idx] += 25
                    pipe_idx += 1

        for idx in range(totalPlayers):
            if playersState[idx] == True:
                if playersVel[idx] < playerMaxVel and not playersFlapped[idx]:
                    playersVel[idx] += playersAcc[idx]
                if playersFlapped[idx]:
                    playersFlapped[idx] = False
                playerHeight = Images['player'][playerIndex].get_height()
                playersYList[idx] += min(playersVel[idx],
                                         BASEY - playersYList[idx] - playerHeight)

        for upperPipe in upperPipes:
            upperPipe['x'] += pipeX

        for lowerPipe in lowerPipes:
            lowerPipe['x'] += pipeX

        if 0 < upperPipes[0]['x'] < 5:
            pipes = getRandomPipe(Images)
            upperPipes.append(pipes[0])
            lowerPipes.append(pipes[1])

        if upperPipes[0]['x'] < -Images['pipes'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        baseX = -((-baseX + 100) % baseShift)
        

        for upperPipe in upperPipes:
            Screen.blit(Images['pipes'][1], (upperPipe['x'], upperPipe['y']))

        for lowerPipe in lowerPipes:
            Screen.blit(Images['pipes'][0], (lowerPipe['x'], lowerPipe['y']))

        for idx in range(totalPlayers):
            if playersState[idx] == True:
                Screen.blit(Images['player'][playerIndex],
                            (playersXList[idx], playersYList[idx]))

        Screen.blit(Images['base'], (baseX, BASEY))
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def nextGeneration(info):
    global fitness, currentPool, generation
    newWeights = []

    generation += 1

    totalFitness = sum(fitness)

    for idx in range(totalPlayers):
        fitness[idx] = fitness[idx]/totalFitness
        if idx > 0:
            fitness[idx] = fitness[idx - 1]

    for idx in range(totalPlayers//2):

        parent1 = random.uniform(0, 1)
        parent2 = random.uniform(0, 1)

        idx1 = -1
        idx2 = -1

        for idxx in range(totalPlayers):
            if fitness[idxx] >= parent1:
                idx1 = idxx
                break

        for idxx in range(totalPlayers):
            if fitness[idxx] >= parent2:
                idx2 = idxx
                break

        newWeights1 = modelCrossover(currentPool, idx1, idx2)
        updatedWeights1 = modelMutate(newWeights1[0])
        updatedWeights2 = modelMutate(newWeights1[1])

        newWeights.append(updatedWeights1)
        newWeights.append(updatedWeights2)

    for i in range(totalPlayers):
        fitness[i] = -100
        currentPool[i].set_weights(newWeights[i])

    return


main()
