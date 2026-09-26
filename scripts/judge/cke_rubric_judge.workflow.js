export const meta = {
  name: 'cke-rubric-judge',
  description: 'Grade an exam answer set item-by-item against the CKE marking scheme (LLM judge, organizer-style), returning per-item points',
  phases: [{ title: 'Judge', detail: 'item groups graded in parallel, essay separately' }],
}

const PY = args.python || 'python3'
const PACKETS = args.packets
const LABEL = args.label || 'answers'
const GROUPS = args.groups || [
  ['1', '2.1', '2.2', '3', '4.1', '4.2', '5.1'],
  ['5.2', '5.3', '6', '7', '8', '9.1', '9.2'],
  ['9.3', '10', '11.1', '11.2', '12', '13.1', '13.2'],
  ['14.1', '14.2', '15', '16.1', '16.2', '17', '18'],
  ['19', '20', '21', '22', '23', '24', '25.1', '25.2'],
  ['26'],
]

const SCHEMA = {
  type: 'object',
  properties: {
    scores: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          points: { type: 'number' },
          max_points: { type: 'number' },
          reason: { type: 'string' },
        },
        required: ['id', 'points', 'max_points', 'reason'],
      },
    },
  },
  required: ['scores'],
}

const show = (ids) => `${PY} -c "import json,sys; ids=set(sys.argv[1:]); [print('=== ITEM', p['id'], '| max', p['max_points'], '| answered', p['answered'], '\\nQUESTION:\\n'+p['question'], '\\nSOURCE (may be truncated):\\n'+(p['source_text'] or '-'), '\\nMARKING SCHEME:\\n'+p['rubric'], '\\nCANDIDATE ANSWER:\\n'+str(p['answer']), '\\n') for p in map(json.loads, open('${PACKETS}')) if p['id'] in ids]" ${ids.map(i => "'" + i + "'").join(' ')}`

phase('Judge')
const results = await parallel(GROUPS.map(ids => () => agent(`You are an experienced CKE examiner grading a Polish HISTORY matura (poziom rozszerzony, May 2023) answer set produced by a language model ("${LABEL}"). Grade exactly like the organizers' "AI rubric assessment": apply the official marking scheme (Zasady oceniania) for each item, award only whole points allowed by the scheme, accept every substantively correct answer that meets the task's conditions ("Akceptowane są wszystkie odpowiedzi merytorycznie poprawne i spełniające warunki zadania"), give 0 for incomplete, wrong, self-contradictory or hedged answers (several alternatives where one is wrong), and 0 for an unanswered item. Closed items: compare with the key exactly (partial credit only where the scheme defines it). Open items: every required element (rozstrzygnięcie, uzasadnienie with reference to the source when required, the requested number of arguments/examples) must be present and correct. The essay (item 26): assess only the first explicitly selected topic; up to 12 points for historical argumentation per the scheme's criteria and up to 3 for coherence; fewer than 300 words = 0 coherence points; follow the scheme's level descriptors.
Print the items with:
  ${show(ids)}
Grade every listed id (${ids.join(', ')}); set points to 0 for unanswered items. Be strict and consistent; one short Polish or English reason per item citing the scheme element that was met or missed.`, { label: `judge:${LABEL}:${ids[0]}-${ids[ids.length - 1]}`, phase: 'Judge', schema: SCHEMA })))

const scores = results.filter(Boolean).flatMap(r => r.scores)
const total = scores.reduce((a, s) => a + (s.points || 0), 0)
log(`${LABEL}: ${scores.length} items graded, total ${total}`)
return { label: LABEL, total, scores }