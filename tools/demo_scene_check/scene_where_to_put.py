#!/usr/bin/env python3
"""Scene "where to put it": ten photos of desks, tables and shelves, 30 placement requests and 23 reference objects.

A scene file is plain data that the other scripts load by path (--scene). Everything in it is fixed before any model
answers. Coordinates are in 0-1000 of the picture (x to the right, y down), read on zoomed grids of the source picture
and checked by drawing every polygon over the photo (fixtures.py write draws geom/<photo>.png) before any model ran.

NAME               = the scene's name (written into fixtures.json).
PHOTOS[photo]      = one Wikimedia Commons file: title, page, original URL, author, licence (CC0 or public domain only,
                     read from the file page's LicenseShortName through the Commons API on 2026-10-31), the original's
                     sha1 (Commons' value; fixtures.py refuses a download with another sha1) and size, a scene word.
SURF[photo][name]  = the polygon of a surface things can be put on (its visible top).
OBJ[photo][name]   = every object standing on (or hanging over) those surfaces: "polys" = its visible outline (an
                     obstacle, and the truth of an object question), "base" = the part that stands on the surface (a
                     lamp's foot, a plant's pot, a pen holder's cup; default = the outline), used to draw the zone.
ITEMS              = the 30 requests. rel_kind lr / fb / between / surface; "zone" = the area the relation names, drawn
                     by hand with one rule: lr = from A's side to the next object on that side or the surface's edge,
                     across A's base rows +- 25 % of its height; fb = from A's front (back) to the next object or the
                     edge, across A's columns +- 25 % of its width; between = the quadrilateral joining A's and B's
                     facing sides; a surface request has no zone (the zone = the surface). The truth region = zone AND
                     (the item's surface shrunk by 4 units) AND NOT (every other object grown by 4 units)
                     (regions.py). "refs" = the objects the request names, with the phrase a detector gets
                     ("det_query") and, for a name with two of a kind, the side to pick; "llm_refs" = the other objects
                     of the photo whose boxes the detector-to-text arm also lists; "rule_solvable" = lr and fb (one
                     reference object, a fixed shift can reach the zone), between and surface = no.
OBJECTS            = the reference objects asked alone ("Point to <name>."): (photo, key in OBJ or "surf:<name>" in
                     SURF, the name in the question, keys subtracted from the outline).
"""

NAME = "where to put it (10 photos, 30 placement requests, 23 reference objects)"

PHOTOS = {
    'p01': {"title": 'File:Minimal desk and painting (Unsplash).jpg', "page": 'https://commons.wikimedia.org/wiki/File:Minimal_desk_and_painting_(Unsplash).jpg',
            "url": 'https://upload.wikimedia.org/wikipedia/commons/2/23/Minimal_desk_and_painting_%28Unsplash%29.jpg',
            "author": 'Breather breather', "license": 'CC0', "usage": 'Creative Commons Zero, Public Domain Dedication',
            "credit": 'https://unsplash.com/photos/kG38b7CFzTY archive copy at the Wayback Machine',
            "sha1": 'c10260664e8db92ce5a6b9bcf278eeab56e211c2', "orig_wh": [5644, 3763], "scene": 'desk'},
    'p02': {"title": 'File:Winter Workspace 2 (Unsplash).jpg', "page": 'https://commons.wikimedia.org/wiki/File:Winter_Workspace_2_(Unsplash).jpg',
            "url": 'https://upload.wikimedia.org/wikipedia/commons/9/92/Winter_Workspace_2_%28Unsplash%29.jpg',
            "author": 'Norbert Levajsics levajsics', "license": 'CC0', "usage": 'Creative Commons Zero, Public Domain Dedication',
            "credit": 'https://unsplash.com/photos/BMYQaySauY0 archive copy at the Wayback Machine Image at the Wayback Machine',
            "sha1": 'f35d529874b9d556fe5510cb2feb153971ae6da5', "orig_wh": [6000, 4000], "scene": 'desk'},
    'p03': {"title": 'File:Coffee in Montreal (Unsplash).jpg', "page": 'https://commons.wikimedia.org/wiki/File:Coffee_in_Montreal_(Unsplash).jpg',
            "url": 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Coffee_in_Montreal_%28Unsplash%29.jpg',
            "author": 'Luke Chesser lukechesser', "license": 'CC0', "usage": 'Creative Commons Zero, Public Domain Dedication',
            "credit": 'http://unsplash.com/photos/KR2mdHJ5qMg Image at the Wayback Machine Gallery at the Wayback Machine',
            "sha1": 'e4025851d513f8fcf7a1621490ea285a667a77eb', "orig_wh": [3456, 2304], "scene": 'cafe table'},
    'p04': {"title": 'File:Plants on wooden shelf (Unsplash).jpg', "page": 'https://commons.wikimedia.org/wiki/File:Plants_on_wooden_shelf_(Unsplash).jpg',
            "url": 'https://upload.wikimedia.org/wikipedia/commons/4/4b/Plants_on_wooden_shelf_%28Unsplash%29.jpg',
            "author": 'Anete Lūsiņa anete_lusina', "license": 'CC0', "usage": 'Creative Commons Zero, Public Domain Dedication',
            "credit": 'https://unsplash.com/photos/xobqx_L-1fg archive copy at the Wayback Machine',
            "sha1": '1562af0458ae9c5fb565fdf02350aa25a5a96aed', "orig_wh": [3000, 1996], "scene": 'shelf'},
    'p05': {"title": 'File:Round table in a bright room (Unsplash).jpg', "page": 'https://commons.wikimedia.org/wiki/File:Round_table_in_a_bright_room_(Unsplash).jpg',
            "url": 'https://upload.wikimedia.org/wikipedia/commons/a/a2/Round_table_in_a_bright_room_%28Unsplash%29.jpg',
            "author": 'Nirzar Pangarkar nirzar', "license": 'CC0', "usage": 'Creative Commons Zero, Public Domain Dedication',
            "credit": 'https://unsplash.com/photos/CswKfD546Z8 Image at the Wayback Machine Gallery at the Wayback Machine',
            "sha1": '8a467735760bfb9f30fa8c84f490737799f2bfec', "orig_wh": [5000, 3333], "scene": 'round table and windowsill'},
    'p06': {"title": 'File:Enamel cups on a table (Unsplash).jpg', "page": 'https://commons.wikimedia.org/wiki/File:Enamel_cups_on_a_table_(Unsplash).jpg',
            "url": 'https://upload.wikimedia.org/wikipedia/commons/b/b3/Enamel_cups_on_a_table_%28Unsplash%29.jpg',
            "author": 'André Freitas andrekerygma', "license": 'CC0', "usage": 'Creative Commons Zero, Public Domain Dedication',
            "credit": 'http://unsplash.com/photos/uu5PfAzu0s4 Image at the Wayback Machine Gallery at the Wayback Machine',
            "sha1": 'e3f807356db38b5d7cf080ed74d42c8df88c80fd', "orig_wh": [4368, 2912], "scene": 'table'},
    'p07': {"title": 'File:Gather (Unsplash).jpg', "page": 'https://commons.wikimedia.org/wiki/File:Gather_(Unsplash).jpg',
            "url": 'https://upload.wikimedia.org/wikipedia/commons/c/c8/Gather_%28Unsplash%29.jpg',
            "author": 'Jeff Sheldon ugmonk', "license": 'CC0', "usage": 'Creative Commons Zero, Public Domain Dedication',
            "credit": 'https://unsplash.com/photos/6MT4_Ut8a3Y Image at the Wayback Machine Gallery at the Wayback Machine',
            "sha1": '8316c843e884ed8ad7e20e02172aca749587470c', "orig_wh": [3333, 4548], "scene": 'kitchen counter'},
    'p08': {"title": 'File:Breakfast on a table (Unsplash).jpg', "page": 'https://commons.wikimedia.org/wiki/File:Breakfast_on_a_table_(Unsplash).jpg',
            "url": 'https://upload.wikimedia.org/wikipedia/commons/c/c3/Breakfast_on_a_table_%28Unsplash%29.jpg',
            "author": 'Katerina Pavlickova katerinapavlickova', "license": 'CC0', "usage": 'Creative Commons Zero, Public Domain Dedication',
            "credit": 'https://unsplash.com/photos/rZPFvJGRaXM Image at the Wayback Machine Gallery at the Wayback Machine',
            "sha1": '23b4d5f26bbde10af61f8e35185fadaf4094b0a4', "orig_wh": [2912, 4368], "scene": 'breakfast table'},
    'p09': {"title": 'File:Bedside (Unsplash).jpg', "page": 'https://commons.wikimedia.org/wiki/File:Bedside_(Unsplash).jpg',
            "url": 'https://upload.wikimedia.org/wikipedia/commons/3/30/Bedside_%28Unsplash%29.jpg',
            "author": 'Rhema Kallianpur rhemakallianpur', "license": 'CC0', "usage": 'Creative Commons Zero, Public Domain Dedication',
            "credit": 'https://unsplash.com/photos/uocSnWMhnAs archive copy at the Wayback Machine',
            "sha1": '0cfc07affa14a6ad17482c3270e2df8ec9f9c06a', "orig_wh": [4096, 2730], "scene": 'nightstand and bed'},
    'p10': {"title": 'File:CleancubicleSouth20050109.jpg', "page": 'https://commons.wikimedia.org/wiki/File:CleancubicleSouth20050109.jpg',
            "url": 'https://upload.wikimedia.org/wikipedia/commons/6/63/CleancubicleSouth20050109.jpg',
            "author": 'AlainV at English Wikipedia', "license": 'Public domain', "usage": 'Public domain',
            "credit": 'Transferred from en.wikipedia to Commons by Sreejithk2000 using CommonsHelper.',
            "sha1": '979f28fed643875c2b826b227a1aa679451419fb', "orig_wh": [1500, 900], "scene": 'office cubicle desk'},
}

SURF = {
    'p01': {'desk': [[135, 785], [868, 785], [878, 879], [122, 879]]},
    'p02': {'desk': [[272, 698], [810, 698], [1000, 900], [1000, 966], [0, 966], [0, 880]]},
    'p03': {'table': [[393, 464], [743, 464], [846, 1000], [307, 1000], [0, 929], [0, 651]]},
    'p04': {'shelf': [[0, 650], [160, 650], [160, 570], [1000, 570], [1000, 707], [343, 1000], [0, 1000]]},
    'p05': {'table': [[29, 197], [71, 138], [143, 101], [214, 75], [300, 60], [400, 58], [500, 69], [571, 90],
                      [615, 130], [636, 190], [620, 215], [571, 236], [500, 256], [400, 266], [300, 265], [214, 256],
                      [143, 243], [71, 226]],
            'chair on the right': [[366, 466], [420, 425], [498, 367], [556, 372], [563, 418], [581, 442], [617, 451],
                                   [654, 441], [660, 568], [617, 579], [545, 585], [473, 566], [402, 515], [366, 477]]},
    'p06': {'table': [[155, 872], [270, 700], [330, 600], [385, 525], [430, 440], [1000, 420], [1000, 1000],
                      [640, 1000]]},
    'p07': {'stove': [[0, 172], [530, 178], [628, 330], [0, 497]],
            'counter': [[0, 497], [628, 330], [660, 440], [1000, 440], [1000, 1000], [330, 1000], [0, 765]]},
    'p08': {'table': [[65, 293], [1000, 297], [1000, 1000], [0, 1000], [0, 350]],
            'saucer': [[100, 612], [330, 545], [600, 520], [720, 535], [968, 700], [960, 735], [400, 832], [330, 830],
                       [300, 815]]},
    'p09': {'nightstand': [[155, 695], [423, 692], [417, 737], [129, 740]],
            'bed': [[557, 776], [605, 798], [684, 761], [830, 703], [1000, 688], [1000, 1000], [478, 1000]]},
    'p10': {'desk': [[440, 470], [510, 460], [640, 456], [1000, 452], [1000, 703], [840, 629], [640, 591],
                     [440, 553]]},
}


def _o(*polys, base=None):
  d = {"polys": [list(p) for p in polys]}
  if base:
    d["base"] = [list(p) for p in base]
  return d


OBJ = {
    'p01': {
        'lamp': _o([[180, 545], [320, 545], [320, 602], [262, 602], [262, 798], [306, 800], [306, 832], [211, 832],
                    [211, 800], [253, 798], [253, 602], [180, 602]],
                   base=[[[211, 798], [306, 798], [306, 832], [211, 832]]]),
        'books': _o([[447, 729], [561, 729], [563, 825], [447, 825]]),
        'brass objects': _o([[663, 733], [763, 733], [763, 828], [663, 828]]),
    },
    'p02': {
        'keyboard': _o([[257, 744], [397, 742], [397, 818], [257, 818]]),
        'mouse': _o([[234, 823], [361, 823], [361, 861], [234, 861]]),
        'pen holder': _o([[575, 580], [590, 580], [590, 798], [607, 798], [607, 885], [559, 885], [559, 798],
                          [575, 798]], base=[[[559, 798], [607, 798], [607, 885], [559, 885]]]),
        'monitor': _o([[588, 100], [625, 108], [565, 722], [500, 722]],
                      [[600, 380], [625, 380], [690, 640], [692, 771], [660, 771], [600, 500]],
                      [[521, 771], [690, 771], [690, 811], [521, 811]]),
        'white cube': _o([[620, 480], [695, 480], [695, 720], [620, 720]]),
        'box with the wooden lid': _o([[706, 797], [861, 797], [861, 928], [706, 928]]),
        'cable': _o([[690, 630], [760, 655], [835, 785], [825, 795], [750, 680], [690, 650]]),
        'radiator': _o([[904, 783], [1000, 783], [1000, 927], [904, 927]]),
    },
    'p03': {
        'phone': _o([[334, 796], [411, 787], [454, 894], [357, 903]]),
        'two cups': _o([[616, 587], [686, 587], [690, 640], [714, 665], [714, 690], [690, 700], [610, 700], [586, 690],
                        [586, 665], [612, 645]],
                       [[593, 641], [704, 641], [706, 700], [712, 725], [712, 780], [680, 798], [590, 798], [555, 780],
                        [555, 735], [590, 719]]),
    },
    'p04': {
        'left plant': _o([[0, 130], [150, 60], [300, 40], [420, 90], [480, 200], [520, 330], [520, 470], [480, 520],
                          [450, 560], [382, 530], [350, 911], [189, 911], [161, 560], [150, 600], [120, 600], [60, 520],
                          [0, 480]], base=[[[161, 530], [382, 530], [350, 911], [189, 911]]]),
        'salt shaker': _o([[361, 668], [414, 668], [414, 830], [361, 830]]),
        'pepper shaker': _o([[516, 659], [596, 659], [596, 827], [516, 827]]),
        'plant on the right': _o([[520, 420], [560, 330], [600, 240], [700, 175], [810, 185], [880, 280], [890, 400],
                                  [860, 450], [805, 480], [800, 766], [646, 766], [624, 600], [593, 592], [564, 592],
                                  [545, 500]], base=[[[624, 493], [800, 493], [771, 766], [646, 766]]]),
    },
    'p05': {
        'books': _o([[249, 136], [420, 120], [454, 229], [264, 245]]),   # the two books with the notebook under them
        'plant in a glass jar': _o([[200, 27], [257, 27], [257, 138], [200, 138]]),
        'bottle': _o([[262, 0], [302, 0], [302, 117], [262, 117]]),
        'chair on the right': _o([[366, 466], [498, 367], [556, 330], [570, 250], [620, 190], [745, 145], [750, 175],
                                  [730, 300], [722, 450], [705, 560], [735, 820], [712, 825], [680, 640], [620, 620],
                                  [612, 985], [585, 985], [575, 620], [460, 620], [400, 830], [380, 830], [440, 600],
                                  [366, 480]]),
    },
    'p06': {
        'white cup at the front': _o([[430, 690], [450, 668], [495, 662], [540, 668], [555, 690], [548, 760],
                                      [540, 860], [520, 880], [490, 885], [460, 878], [445, 860], [438, 760]]),
        'red cup front': _o([[360, 680], [380, 660], [420, 658], [440, 675], [440, 790], [415, 808], [375, 805],
                             [345, 790], [345, 730]]),
        'red cup left': _o([[312, 625], [330, 607], [380, 605], [395, 620], [392, 720], [370, 730], [330, 725],
                            [310, 680]]),
        'red cup back': _o([[395, 560], [410, 545], [455, 543], [470, 560], [468, 640], [440, 660], [400, 655],
                            [395, 600]]),
        'red cup middle': _o([[472, 600], [490, 588], [530, 588], [545, 600], [543, 660], [510, 672], [475, 665]]),
        'white cup middle': _o([[560, 610], [580, 598], [640, 596], [668, 610], [662, 760], [640, 782], [600, 785],
                                [575, 770], [562, 700]]),
        'white cup back': _o([[680, 545], [700, 532], [760, 530], [778, 545], [772, 680], [745, 695], [705, 693],
                              [685, 670]]),
        'white cup right': _o([[810, 525], [830, 515], [890, 515], [905, 525], [925, 580], [915, 640], [880, 665],
                               [830, 660], [812, 640]]),
        'glass shaker': _o([[835, 640], [855, 625], [895, 625], [912, 640], [910, 840], [890, 850], [850, 850],
                            [832, 835]]),
        'white cloth': _o([[385, 395], [420, 375], [560, 365], [640, 380], [720, 440], [760, 520], [700, 560],
                           [610, 570], [560, 560], [470, 540], [400, 480]]),
        'plate': _o([[965, 500], [1000, 490], [1000, 585], [970, 580]]),
    },
    'p07': {
        'stove': _o([[0, 172], [530, 178], [628, 330], [0, 497]]),
        'pot': _o([[0, 205], [150, 205], [230, 215], [300, 245], [290, 300], [200, 330], [150, 400], [0, 420]]),
        'tablet': _o([[490, 328], [705, 385], [548, 714], [372, 714], [370, 660], [340, 620]]),
        'left containers': _o([[182, 560], [260, 530], [386, 523], [400, 560], [386, 629], [182, 629]]),
        'ruler': _o([[120, 520], [262, 530], [262, 560], [120, 550]]),
        'tray': _o([[179, 576], [386, 629], [548, 714], [621, 705], [852, 700], [868, 770], [868, 790], [702, 876],
                    [182, 608]]),
        'organizer items': _o([[552, 700], [580, 610], [726, 610], [740, 660], [852, 660], [852, 833], [700, 840],
                               [552, 796]]),
        'spoons': _o([[765, 450], [800, 435], [850, 440], [862, 480], [850, 560], [830, 620], [830, 700], [790, 700],
                      [790, 620], [775, 560], [765, 500]]),
        'bowl of limes': _o([[785, 330], [830, 300], [900, 295], [980, 320], [1000, 330], [1000, 500], [900, 505],
                             [800, 492], [785, 450]]),
        'coffee maker': _o([[620, 100], [860, 100], [860, 330], [800, 330], [760, 440], [640, 440], [620, 330]]),
    },
    'p08': {
        'croissant': _o([[55, 445], [110, 385], [200, 340], [280, 322], [420, 330], [540, 345], [600, 370], [620, 405],
                         [560, 420], [300, 440], [240, 470], [150, 493], [70, 490], [55, 470]]),
        'cup': _o([[237, 430], [300, 412], [480, 408], [650, 412], [720, 440], [715, 540], [690, 640], [660, 700],
                   [600, 740], [480, 755], [360, 740], [305, 700], [280, 640], [262, 560], [245, 500]],
                  [[245, 535], [160, 545], [130, 580], [132, 640], [160, 672], [240, 680], [300, 670], [300, 560]]),
        'glass of water': _o([[700, 355], [760, 347], [900, 347], [930, 365], [905, 420], [890, 500], [860, 530],
                              [770, 535], [740, 500], [720, 420], [700, 380]]),
        'saucer': _o([[100, 612], [330, 545], [600, 520], [720, 535], [968, 700], [960, 735], [400, 832], [330, 830],
                      [300, 815]]),
    },
    'p09': {
        'lamp': _o([[237, 60], [360, 60], [360, 315], [305, 315], [305, 682], [354, 682], [354, 705], [254, 705],
                    [254, 682], [296, 682], [296, 315], [237, 315]], base=[[[254, 682], [354, 682], [354, 705],
                                                                           [254, 705]]]),
        'bottle': _o([[380, 440], [400, 440], [405, 480], [412, 560], [412, 699], [367, 699], [367, 560], [372, 480]]),
        'nightstand': _o([[125, 690], [425, 690], [425, 1000], [125, 1000]]),
        'pillows': _o([[601, 310], [1000, 347], [1000, 688], [830, 703], [684, 761], [605, 798], [557, 776],
                       [532, 600], [562, 549], [601, 542]]),
        'bed': _o([[557, 776], [605, 798], [684, 761], [830, 703], [1000, 688], [1000, 1000], [478, 1000]],
                  [[601, 310], [1000, 347], [1000, 688], [830, 703], [684, 761], [605, 798], [557, 776], [532, 600],
                   [562, 549], [601, 542]],
                  [[485, 319], [1000, 319], [1000, 1000], [478, 1000]]),   # duvet, pillows, headboard
    },
    'p10': {
        'monitor': _o([[356, 340], [470, 330], [486, 345], [486, 520], [483, 543], [440, 552], [390, 552], [372, 525],
                       [356, 420]]),
        'printer': _o([[510, 355], [637, 355], [637, 510], [611, 529], [510, 529]]),
        'pen cup': _o([[629, 455], [670, 455], [670, 535], [650, 543], [629, 540]]),
        'purple folder': _o([[651, 569], [709, 493], [764, 515], [701, 593]]),
        'document holder': _o([[866, 420], [1000, 380], [1000, 500], [880, 526]]),
    },
}


def _ref(name, det, pick=None):
  r = {"name": name, "det_query": det}
  if pick:
    r["pick"] = pick
  return r


def _item(iid, request, rel_kind, rel, refs, surface, zone, llm_refs, note=""):
  return {"id": iid, "photo": iid[:3], "request": request, "rel_kind": rel_kind, "rel": rel, "refs": refs,
          "surface": surface, "zone": zone, "llm_refs": [{"det_query": q} for q in llm_refs],
          "rule_solvable": rel_kind in ("lr", "fb"), "note": note}


# detector-to-text arm: the boxes of the reference objects, then of these other objects of the photo
_CTX = {
    'p01': ["brass sculpture"],
    'p02': ["computer monitor", "computer mouse"],
    'p03': [],
    'p04': [],
    'p05': ["bottle", "table"],
    'p06': ["red cup"],
    'p07': ["tablet", "tray", "wooden spoon", "pot"],
    'p08': ["cup", "croissant", "glass of water"],
    'p09': ["bottle", "pillow"],
    'p10': ["pen holder", "computer monitor"],
}

ITEMS = [
    _item('p01_right_books', "Put the mug to the right of the books.", "lr", "right", [_ref("books", "books")],
          "desk", [[563, 705], [663, 705], [663, 849], [563, 849]], _CTX['p01'] + ["lamp"]),
    _item('p01_between_lamp_books', "Put the mug between the lamp and the books.", "between", "between",
          [_ref("lamp", "lamp"), _ref("books", "books")], "desk",
          [[306, 798], [447, 729], [447, 825], [306, 832]], _CTX['p01']),
    _item('p01_front_books', "Put the mug in front of the books.", "fb", "front", [_ref("books", "books")],
          "desk", [[418, 825], [592, 825], [592, 1000], [418, 1000]], _CTX['p01'] + ["lamp"]),
    _item('p02_between_keyboard_penholder', "Put the phone between the keyboard and the pen holder.", "between",
          "between", [_ref("keyboard", "keyboard"), _ref("pen holder", "pen holder")], "desk",
          [[397, 742], [559, 798], [559, 885], [397, 818]], _CTX['p02'] + ["box with a wooden lid"]),
    _item('p02_right_penholder', "Put the phone to the right of the pen holder.", "lr", "right",
          [_ref("pen holder", "pen holder")], "desk", [[607, 776], [706, 776], [706, 907], [607, 907]],
          _CTX['p02'] + ["keyboard", "box with a wooden lid"]),
    _item('p02_front_box', "Put the phone in front of the box with the wooden lid.", "fb", "front",
          [_ref("box with the wooden lid", "box with a wooden lid")], "desk",
          [[667, 928], [900, 928], [900, 1000], [667, 1000]], _CTX['p02'] + ["keyboard", "pen holder"]),
    _item('p03_behind_cups', "Put the glass behind the two cups.", "fb", "behind", [_ref("two cups", "cup")],
          "table", [[515, 0], [754, 0], [754, 587], [515, 587]], ["phone"]),
    _item('p03_left_phone', "Put the glass to the left of the phone.", "lr", "left", [_ref("phone", "phone")],
          "table", [[0, 758], [334, 758], [334, 932], [0, 932]], ["cup"]),
    _item('p03_between_phone_cups', "Put the glass between the phone and the cups.", "between", "between",
          [_ref("phone", "phone"), _ref("two cups", "cup")], "table",
          [[411, 787], [555, 587], [555, 798], [454, 894]], []),
    _item('p04_between_salt_pepper', "Put the small jar between the salt shaker and the pepper shaker.", "between",
          "between", [_ref("salt shaker", "salt shaker"), _ref("pepper shaker", "pepper shaker")], "shelf",
          [[414, 668], [516, 659], [516, 827], [414, 830]], ["potted plant"]),
    _item('p04_front_pepper', "Put the small jar in front of the pepper shaker.", "fb", "front",
          [_ref("pepper shaker", "pepper shaker")], "shelf", [[496, 827], [616, 827], [616, 1000], [496, 1000]],
          ["salt shaker", "potted plant"]),
    _item('p04_between_pepper_plant', "Put the small jar between the pepper shaker and the plant on the right.",
          "between", "between", [_ref("pepper shaker", "pepper shaker"),
                                 _ref("plant on the right", "potted plant", "right")], "shelf",
          [[596, 659], [624, 493], [646, 766], [596, 827]], ["salt shaker"]),
    _item('p05_right_books', "Put the vase to the right of the books.", "lr", "right", [_ref("books", "books")],
          "table", [[454, 89], [1000, 89], [1000, 276], [454, 276]], _CTX['p05'] + ["plant in a glass jar"]),
    _item('p05_behind_books', "Put the vase behind the books.", "fb", "behind", [_ref("books", "books")], "table",
          [[198, 0], [505, 0], [505, 125], [454, 125], [420, 120], [249, 136], [198, 136]],
          _CTX['p05'] + ["plant in a glass jar"]),
    _item('p05_on_chair', "Put the bag on the chair on the right.", "surface", "on",
          [_ref("chair on the right", "chair", "right")], "chair on the right", None, ["table"]),
    _item('p06_front_shaker', "Put the sugar bowl in front of the glass shaker.", "fb", "front",
          [_ref("glass shaker", "glass shaker")], "table", [[809, 850], [936, 850], [936, 1000], [809, 1000]],
          _CTX['p06'] + ["white cup"]),
    _item('p06_between_cup_shaker', "Put the sugar bowl between the white cup at the front and the glass shaker.",
          "between", "between", [_ref("white cup at the front", "white cup", "front"),
                                 _ref("glass shaker", "glass shaker")], "table",
          [[555, 665], [832, 625], [832, 850], [555, 885]], _CTX['p06']),
    _item('p06_right_shaker', "Put the sugar bowl to the right of the glass shaker.", "lr", "right",
          [_ref("glass shaker", "glass shaker")], "table", [[912, 569], [1000, 569], [1000, 906], [912, 906]],
          _CTX['p06'] + ["white cup"]),
    _item('p07_on_stove', "Put the pan on the stove.", "surface", "on", [_ref("stove", "stove")], "stove", None,
          ["pot", "counter"]),
    _item('p07_on_counter', "Put the cup on the counter.", "surface", "on", [_ref("counter", "counter")], "counter",
          None, _CTX['p07'] + ["stove", "bowl of limes"]),
    _item('p07_front_limes', "Put the cup in front of the bowl of limes.", "fb", "front",
          [_ref("bowl of limes", "bowl of limes")], "counter",
          [[731, 500], [1000, 500], [1000, 1000], [868, 1000], [868, 610], [731, 610]], _CTX['p07']),
    _item('p08_on_saucer', "Put the spoon on the saucer.", "surface", "on", [_ref("saucer", "saucer")], "saucer",
          None, _CTX['p08']),
    _item('p08_front_saucer', "Put the spoon in front of the saucer.", "fb", "front", [_ref("saucer", "saucer")],
          "table", [[0, 832], [100, 832], [100, 612], [330, 830], [400, 832], [960, 735], [968, 700], [968, 832],
                    [1000, 832], [1000, 1000], [0, 1000]], _CTX['p08']),
    _item('p08_left_saucer', "Put the spoon to the left of the saucer.", "lr", "left", [_ref("saucer", "saucer")],
          "table", [[0, 442], [100, 442], [100, 612], [330, 830], [100, 830], [100, 910], [0, 910]], _CTX['p08']),
    _item('p09_on_nightstand', "Put the book on the nightstand.", "surface", "on",
          [_ref("nightstand", "nightstand")], "nightstand", None, _CTX['p09'] + ["lamp", "bed"]),
    _item('p09_on_bed', "Put the book on the bed.", "surface", "on", [_ref("bed", "bed")], "bed", None,
          _CTX['p09'] + ["lamp", "nightstand"]),
    _item('p09_left_lamp', "Put the glass to the left of the lamp.", "lr", "left", [_ref("lamp", "lamp")],
          "nightstand", [[100, 640], [254, 640], [254, 760], [100, 760]], _CTX['p09'] + ["nightstand"],
          note="the nightstand top is seen almost edge-on (about 45 units deep), so the zone spans its whole depth "
               "instead of the lamp foot's rows +- 25 % (6 units)"),
    _item('p10_right_folder', "Put the mug to the right of the purple folder.", "lr", "right",
          [_ref("purple folder", "purple folder")], "desk",
          [[764, 468], [866, 468], [866, 619], [764, 619], [764, 593], [701, 593], [764, 515]],
          _CTX['p10'] + ["printer"]),
    _item('p10_behind_folder', "Put the mug behind the purple folder.", "fb", "behind",
          [_ref("purple folder", "purple folder")], "desk",
          [[623, 0], [792, 0], [792, 515], [764, 515], [709, 493], [651, 569], [623, 569]], _CTX['p10'] + ["printer"]),
    _item('p10_front_printer', "Put the mug in front of the printer.", "fb", "front", [_ref("printer", "printer")],
          "desk", [[478, 529], [669, 529], [669, 1000], [478, 1000]], _CTX['p10'] + ["purple folder"]),
]

# reference objects asked alone: (photo, key in OBJ (or SURF when "surf:"), name in the prompt, keys subtracted)
OBJECTS = [
    ('p01', 'books', 'the books', []), ('p01', 'lamp', 'the lamp', []),
    ('p02', 'keyboard', 'the keyboard', []), ('p02', 'pen holder', 'the pen holder', []),
    ('p02', 'box with the wooden lid', 'the box with the wooden lid', []),
    ('p03', 'phone', 'the phone', []), ('p03', 'two cups', 'the two cups', []),
    ('p04', 'salt shaker', 'the salt shaker', []), ('p04', 'pepper shaker', 'the pepper shaker', []),
    ('p04', 'plant on the right', 'the plant on the right', []),
    ('p05', 'books', 'the books', []), ('p05', 'chair on the right', 'the chair on the right', []),
    ('p06', 'glass shaker', 'the glass shaker', []), ('p06', 'white cup at the front', 'the white cup at the front', []),
    ('p07', 'stove', 'the stove', ['pot']),
    ('p07', 'surf:counter', 'the counter', ['pot', 'tablet', 'left containers', 'ruler', 'tray', 'organizer items',
                                            'spoons', 'bowl of limes', 'coffee maker']),
    ('p07', 'bowl of limes', 'the bowl of limes', []),
    ('p08', 'saucer', 'the saucer', ['cup']),
    ('p09', 'nightstand', 'the nightstand', ['lamp', 'bottle']), ('p09', 'bed', 'the bed', []),
    ('p09', 'lamp', 'the lamp', []),
    ('p10', 'printer', 'the printer', []), ('p10', 'purple folder', 'the purple folder', []),
]
