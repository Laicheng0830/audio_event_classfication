"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/5/8 17:40
"""

def config(game):
    if game == 'pubg':
        # PUBG CONFIG
        # GUN = 'F:/Audio_dataset/games_audio/gun/'
        # FOOTSTEP = 'F:/Audio_dataset/games_audio/footstep/'
        # VEHICLE = 'F:/Audio_dataset/games_audio/vehicle/'
        # GUN_FOOTSTEP = 'F:/Audio_dataset/games_audio/gun_footstep/'
        # GUN_VEHICLE = 'F:/Audio_dataset/games_audio/gun_vehicle/'
        # FOOTSTEP_GUN = 'F:/Audio_dataset/games_audio/footstep_gun/'
        # FOOTSTEP_VEHICLE = 'F:/Audio_dataset/games_audio/footstep_vehicle/'
        # VEHICLE_GUN = 'F:/Audio_dataset/games_audio/vehicle_gun/'
        # VEHICLE_FOOTSTEP = 'F:/Audio_dataset/games_audio/vehicle_footstep/'
        # OTHER = 'F:/Audio_dataset/games_audio/other/'


        #  get test dataset
        GUN = 'F:/Audio_dataset/games_audio_test/gun/'
        FOOTSTEP = 'F:/Audio_dataset/games_audio_test/footstep/'
        VEHICLE = 'F:/Audio_dataset/games_audio_test/vehicle/'
        GUN_FOOTSTEP = 'F:/Audio_dataset/games_audio_test/gun_footstep/'
        GUN_VEHICLE = 'F:/Audio_dataset/games_audio_test/gun_vehicle/'
        FOOTSTEP_GUN = 'F:/Audio_dataset/games_audio_test/footstep_gun/'
        FOOTSTEP_VEHICLE = 'F:/Audio_dataset/games_audio_test/footstep_vehicle/'
        VEHICLE_GUN = 'F:/Audio_dataset/games_audio_test/vehicle_gun/'
        VEHICLE_FOOTSTEP = 'F:/Audio_dataset/games_audio_test/vehicle_footstep/'
        OTHER = 'F:/Audio_dataset/games_audio_test/other/'
        DIR = [GUN, FOOTSTEP, VEHICLE, GUN_FOOTSTEP, GUN_VEHICLE, FOOTSTEP_GUN,
               FOOTSTEP_VEHICLE, VEHICLE_GUN, VEHICLE_FOOTSTEP,OTHER]

    if game == 'fortnite':
        # FORTNITE CONFIG
        GUN = ''
        FOOTSTEP = ''
        CHEST = ''
        HNOCK = ''
        GUN_FOOTSTEP = ''
        GUN_CHEST = ''
        GUN_HNOCK = ''
        FOOTSTEP_GUN = ''
        FOOTSTEP_CHEST = ''
        FOOTSTEP_HNOCK = ''
        CHEST_GUN = ''
        CHEST_FOOTSTEP = ''
        CHEST_HNOCK = ''
        HNOCK_GUN = ''
        HNOCK_FOOTSTEP = ''
        HNOCK_CHEST = ''
        OTHER = ''
        DIR = [GUN, FOOTSTEP, CHEST, HNOCK, GUN_FOOTSTEP, GUN_CHEST, GUN_HNOCK, FOOTSTEP_GUN,
               FOOTSTEP_CHEST, FOOTSTEP_HNOCK, CHEST_GUN, CHEST_FOOTSTEP, CHEST_HNOCK, HNOCK_GUN,
               HNOCK_FOOTSTEP, HNOCK_CHEST, OTHER]

    return DIR
