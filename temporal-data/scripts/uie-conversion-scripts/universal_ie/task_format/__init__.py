#!/usr/bin/env python
# -*- coding:utf-8 -*-
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.task_format.oneie import OneIEEvent
from universal_ie.task_format.jointer import JointER
from universal_ie.task_format.mrc_ner import MRCNER
from universal_ie.task_format.absa import ABSA
from universal_ie.task_format.pate import PATE
from universal_ie.task_format.snips import SNIPS
from universal_ie.task_format.fullpate import FULLPATE
from universal_ie.task_format.timebank import TIMEBANK
from universal_ie.task_format.aquaint import AQUAINT
from universal_ie.task_format.tempeval import TEMPEVAL
from universal_ie.task_format.tempeval import TEMPEVALENTITY
from universal_ie.task_format.wikiwars import WIKIWARS
from universal_ie.task_format.wikiwarstagged import WIKIWARSTAGGED
from universal_ie.task_format.tweets import TWEETS
from universal_ie.task_format.spannet import Spannet
from universal_ie.task_format.casie import CASIE
from universal_ie.task_format.tempevalrelation import TEMPEVALRELATION
from universal_ie.task_format.cols import (
    TokenTagCols,
    I2b2Conll,
    TagTokenCols,
    TokenTagJson,
    CoNLL03,
)
