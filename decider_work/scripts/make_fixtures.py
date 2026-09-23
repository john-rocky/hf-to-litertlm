"""Freeze the prescribed synthetic fixture mix before any inference."""
from collections import Counter
from transformers import AutoTokenizer
from decider.systemone import render_state, render_question, plan_rows
from decider.prompt import build, MAX_OPTIONS
from decider.infer import Example, Q
from common import ROOT, snapshot_path, write_json, sha256


class Keep:
    def shuffle(self, x):
        pass
    def sample(self, xs, k):
        return xs[:k]


def main():
    tok = AutoTokenizer.from_pretrained(str(snapshot_path()), local_files_only=True)
    fixtures = []

    def add(family, state, questions):
        i = sum(f['family'] == family for f in fixtures) + 1
        fixtures.append(dict(id=f'{family}{i:02}', family=family, state=state, questions=questions))

    tickets = [
        'I was charged twice for order ZVX-104. Please refund the duplicate payment. The first payment was correct.',
        'The invoice lists an extra storage fee. Please explain it. I am not requesting a refund at this time.',
        'The export screen freezes whenever I open a saved report. I have retried four times today and cannot finish my work.',
        'Could someone explain the larger plan for a small workshop? We need six seats next month. Everything currently works.',
        'My parcel has not arrived although the tracking page says delivered. Please investigate the shipping record.',
        'I cannot reset my password because the recovery code is rejected. I am losing patience after three failed attempts.',
        'Please return the payment for the cancelled booking. Two previous replies ignored that request. This is very frustrating.',
        'Thank you for fixing the display problem yesterday. I would like to suggest a darker border around the status icon.',
        'The service was unavailable through the whole afternoon. I need the saved work restored, and I want a refund for the lost day.',
        'The replacement cable works well. Could the package include a shorter printed care guide next time?',
    ]
    all_teams = {
        'billing':'Charges, invoices, payments, refunds',
        'technical':'Software errors, outages, broken features',
        'accounts':'Login, passwords, account access',
        'shipping':'Parcels, delivery, missing shipments',
        'plans':'New plans, seats, prices before purchase',
    }
    for i, ticket in enumerate(tickets):
        criteria = dict(list(all_teams.items())[:3 + i % 3])
        criteria['other'] = None
        add('A', dict(ticket=ticket, customer_tier=['basic','plus','premium'][i % 3],
                      history=[dict(step=1, event='Request received'),
                               dict(step=2, event=['No prior complaint','One previous reply','Follow-up requested'][i % 3])]),
            dict(team=dict(type='choice',instructions='Which team should handle this ticket?',criteria=criteria),
                 refund_requested=dict(type='noul',instructions='Does the customer ask for a refund?'),
                 frustration=dict(type='score',instructions='How frustrated is the customer?',
                                  criteria=['calm','frustrated','very frustrated'])))

    commands = [
        ('Pause the music.', ['play','pause','stop','repeat']),
        ('Make the room brighter.', ['dim','brighten','cool','warm','switch off']),
        ('Set a timer for nine minutes.', ['timer','alarm','calendar','weather','volume','search']),
        ('Please close the window.', ['open window','close window','open door','close door','lock door','unlock door','leave unchanged']),
        ('I need directions to the river path.', ['navigation','translate','calculate','weather','music','timer','notes','lights']),
        ('Save this thought as a note.', ['save note','delete note','read note','find file','rename file','copy text','paste text','open folder','close folder']),
        ('Turn the fan off.', ['fan on','fan off','fan faster','fan slower','heater on','heater off','light on','light off','curtain open','curtain close']),
        ('What is six multiplied by seven?', ['calculate','spell','translate','navigate']),
    ]
    for state, options in commands:
        add('B', state, {'intent':dict(type='choice',instructions='Which intent best matches the request?',criteria=options)})

    for i in range(6):
        state = dict(record=dict(code=f'ZRV-{230+i}', approved=i%2==0,
                                 shipment=dict(status=['ready','held','dispatched'][i%3], inspected=i%3!=1)),
                     audit=dict(reviewed=True, missing_fields=[] if i%2==0 else ['seal']))
        if i in (4,5):
            state['records'] = [dict(code=f'VQX-{j+410}', sealed=j != i, temperature=18+j) for j in range(8+i-4)]
            state['record']['target_position'] = i
        add('C',state,dict(approved=dict(type='noul',instructions='Is record.approved true?',
                                            criteria={'true':'The approved field is true','false':'The approved field is false'}),
                           inspected=dict(type='noul',instructions='Has record.shipment been inspected?')))

    score_sets = [
        ['unfinished','finished'],
        ['calm','concerned','panicked'],
        ['no delay','short delay','long delay','work stopped'],
        ['unusable','poor','adequate','good','excellent'],
        ['none','trace','slight','moderate','substantial','severe','complete'],
        ['empty','almost empty','very little','a little','below half','about half','above half','mostly full','nearly full','completely full'],
    ]
    score_questions = ['How complete is the assembly?', 'How worried is the speaker?', 'How large is the delay?',
                       'How good is the workmanship?', 'How much damage is described?', 'How full is the storage tank?']
    score_states = [
        'The assembly has passed every check on the work card. All panels are fastened and the cover is fitted. The inspector signed the final box this morning. There are no parts left on the tray, no open tasks in the log, and no further work is planned before packing.',
        'The speaker says the unusual noise is worth checking but there is still time to inspect it. They have arranged a careful review after lunch and asked the team to keep notes. Their voice is steady. They are worried about a possible fault, yet there is no immediate danger or urgent evacuation.',
        'The scheduled departure was held up for twenty minutes while a gate was cleared. The driver then resumed the usual route. Two deliveries arrived a little late, but the afternoon schedule recovered. Work continued throughout the rest of the day and nobody cancelled an order because of the delay.',
        'The cabinet has even joints and a smooth finish. Every drawer slides easily and the doors line up with the frame. A small scratch is visible inside the bottom compartment, but it does not affect use. The reviewer considers the work good overall and would accept another item made to the same standard.',
        'Water reached the lowest shelf and ruined several paper cartons. The metal frame remains straight and the upper shelves are dry. The room can still be used after cleaning, although a considerable part of the stored paper must be replaced. No wall has collapsed and the main equipment is still working.',
        'The tank gauge points exactly to the middle mark. A manual reading confirms that the liquid reaches half of the marked capacity. No inflow or outflow is active, so the level is stable. The operator records the reading before closing the inspection cover and leaves the remaining volume available for the next delivery.',
    ]
    for state, instruction, levels in zip(score_states,score_questions,score_sets):
        add('D',state,{'assessment':dict(type='score',instructions=instruction,criteria=levels)})

    dialogues = [
        ['I need to move my booking.','Which day would work?','The next morning, please.','I can record that request.'],
        ['The total seems wrong.','Which charge concerns you?','The second service charge.','Was the service used twice?','No, only once.','I will flag the duplicate.'],
        ['The page remains blank.','Did reloading help?','No, it did not.','Does a different page open?','Yes, the help page opens.','I will note the affected page.','Please have someone repair it.','The repair request is recorded.'],
        ['I want to end my subscription.','When should it end?','At the end of this week.','That requested date is noted.'],
        ['Can I buy another seat?','How many seats do you need?','Two more for the workshop.','Will the current plan stay?','Yes, keep the plan.','I will record the seat request.'],
        ['The parcel reached the wrong door.','Is the address label correct?','Yes, the label is correct.','Can you collect it safely?','No, the building is locked.','I will flag the delivery issue.','Please arrange a new delivery.','The request is recorded.'],
    ]
    conversation_options = {'billing':'Payments or incorrect charges','technical':'A broken feature or error',
                            'changes':'A booking, subscription, or seat change','delivery':'Parcels and shipping','other':'Another subject'}
    for dialog in dialogues:
        add('E',[dict(role='user' if i%2==0 else 'assistant',content=s) for i,s in enumerate(dialog)],
            dict(topic=dict(type='choice',instructions='What is the main topic of this conversation?',criteria=conversation_options),
                 action_requested=dict(type='noul',instructions='Does the user ask for a concrete action?')))

    long_topics = [
        ('Incident report',
         'The central relay stopped forwarding sensor readings after a loose power connector vibrated out of its socket. A technician reseated the connector and fitted a retaining clip. The relay resumed normal service, and inspection found no data deletion or unauthorized access.',
         'Inspection phase {i} covered zone {zone}. The checklist recorded the connector position, indicator status, enclosure temperature, and the time of each reading. The reviewer compared the written observations with the previous shift and found no additional fault. The attached note states that the cable route was kept clear during the check.',
         {'power':'A power supply or connector problem','software':'A software logic error','intrusion':'Unauthorized access'},
         'What caused the interruption?', 'Was service restored?'),
        ('Assembly station specification',
         'The proposed assembly station uses a foot pedal to move a sliding clamp. It operates without a network connection. The safety cover must remain shut during motion, and opening it stops the clamp. A manual release allows the operator to remove a stalled item after power is isolated.',
         'Requirement group {i} applies to bench {zone}. The inspection sheet lists the clamp travel, cover clearance, pedal return, and visibility of the status flag. A reviewer must record each result before approving that group. The operator should be able to read the flag from the normal working position without reaching into the moving section.',
         {'mechanical':'A physical assembly workstation','messaging':'A communication service','storage':'A file archive','planning':'A scheduling tool'},
         'What kind of product is specified?', 'Does opening the cover stop motion?'),
        ('Workshop access policy',
         'Visitors may enter the workshop only with a trained escort. They must sign the arrival sheet, wear the issued eye protection, and stay behind the marked line when machinery is moving. An escort may refuse entry if these conditions are not met. Staff performing their assigned duties follow the separate staff procedure.',
         'Review section {i} concerns area {zone}. The escort checks the visible floor markings, the supply of eye protection, and the legibility of the arrival sheet before the visit begins. The local supervisor reviews the written record at the end of the day. Any damaged marking must be reported and replaced before another group enters that area.',
         {'escorted':'Entry with a trained escort and stated precautions','unrestricted':'Entry without an escort or precautions'},
         'Which access rule applies to visitors?', 'Must visitors wear eye protection?'),
    ]
    for title, intro, paragraph, choices, q1, q2 in long_topics:
        state=title+'\n'+intro
        for i in range(1,9):
            state+='\n'+paragraph.format(i=i,zone=['north','south','east','west'][i%4])
            if len(tok.encode(state,add_special_tokens=False)) >= 470:
                break
        add('F',state,dict(category=dict(type='choice',instructions=q1,criteria=choices),
                          condition=dict(type='noul',instructions=q2)))

    codes=[f'ZVQ-{i:03}' for i in range(100,355)]
    add('G','The selected item code is ZVQ-287. Read the code from this sentence.',
        {'selected_code':dict(type='choice',instructions='Which item code is selected?',criteria=codes)})

    assert Counter(f['family'] for f in fixtures)==dict(A=10,B=8,C=6,D=6,E=6,F=3,G=1)
    rows=[]
    for f in fixtures:
        context=render_state(f['state'])
        f['state_tokens']=len(tok.encode(context,add_special_tokens=False))
        if f['family']=='B':
            assert f['state_tokens'] <= 25
        if f['family']=='D':
            assert 40 <= f['state_tokens'] <= 120
        if f['family']=='F':
            assert 400 <= f['state_tokens'] <= 900
        rqs={k:render_question(v) for k,v in f['questions'].items()}
        flat,index=plan_rows(rqs,True)
        for j,r in enumerate(flat):
            item=build(Example(context,[Q(r['question'],list(r['options']),0)]),tok,Keep(),max_options=MAX_OPTIONS,max_ctx_tokens=32768,layout='state_first')
            assert item['slots']==[len(item['ids'])-1]
            assert len(item['ids'])<=4096
            rows.append(dict(fixture_id=f['id'],family=f['family'],row_index=j,n_tokens=len(item['ids']),nopts=item['nopts'][0],
                             token_zero_positions=[p for p,t in enumerate(item['ids']) if t==0]))
    assert len(rows)==120
    assert rows[-1]['nopts']==255 and rows[-1]['n_tokens']>1000
    write_json('fixtures/fixtures.json',fixtures)
    write_json('results/fixture_design.json',dict(fixtures=40,rows=len(rows),family_counts=dict(Counter(f['family'] for f in fixtures)),
                                               rows_design=rows,fixture_sha256=sha256(ROOT/'fixtures/fixtures.json'),
                                               note='Specified fixture composition gives exactly 120 rows (50+8+12+31+12+6+1).'))
    print('Frozen 40 fixtures / 120 rows; SHA256',sha256(ROOT/'fixtures/fixtures.json'),flush=True)
    for f in fixtures:
        print(f['id'],f['state_tokens'],'state tokens',flush=True)


if __name__=='__main__':
    main()
