const SCRIPT_URL = "https://script.google.com/macros/s/AKfycbx8W9uN6Xp_AqJP3syHQh-Civ4khsXKBcyYIjKIAf1g-bz9813tVu__dgHLsedl7WVt/exec";
const EVALUATIONS_CSV_PATH = "../sample_master_annotator_pool.csv";
const QUESTIONS_CSV_PATH = "../prompts.csv";
const BATCH_SIZE = 10;
const PROMPT_TYPES = ['minimum', 'contextual', 'detailed', 'structured'];

let dataCombined = [], samplesAvailable = [], batchCurrent = [], CurrentBatchIndex = 0, humanId = "";
let notesMadeIDs = new Set();

window.addEventListener('DOMContentLoaded', () => {
    generateHumanId();
    loadData();
});

function generateHumanId() {
    humanId = localStorage.getItem('humanId');
    if (!humanId) {
        humanId = 'user-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('humanId', humanId);
    }
    const noted = localStorage.getItem('notesMadeIDs');
    if (noted) notesMadeIDs = new Set(JSON.parse(noted));
}

function showScreen(id) {
    ['instructions-container', 'setup-container', 'annotation-container', 'batch-complete-container', 'all-done-container'].forEach(cId => document.getElementById(cId).classList.add('hidden'));
    document.getElementById(id).classList.remove('hidden');
}

function showSetupScreen() { showScreen('setup-container'); }

function endSession() {
    const container = document.getElementById('batch-complete-container');
    container.innerHTML = '<h2>Obrigado!</h2><p>Sua contribuição foi registrada. Pode fechar esta aba.</p>';
    showScreen('batch-complete-container');
}

async function loadData() {
    try {
        const [evaluations, questions] = await Promise.all([parseCsv(EVALUATIONS_CSV_PATH), parseCsv(QUESTIONS_CSV_PATH)]);
        const questionsMap = new Map(questions.map(q => [q.question_id, q]));

        dataCombined = evaluations.map(ev => {
            const questionRow = questionsMap.get(ev.question_id);
            if (!questionRow) return null;

            let languageValue = ev.language;
            if (!languageValue || languageValue.trim() === '') {
                languageValue = 'en';
            }

            const idParts = ev.evaluation_id.toLowerCase().split('_');
            let promptType = idParts.find(part => PROMPT_TYPES.includes(part));
            if (!promptType) return null;

            let normalizedLanguage = languageValue.split('-')[0].toLowerCase();
            const columnName = `prompt_${promptType}_${normalizedLanguage}`;
            const promptText = questionRow[columnName];
            
            if (!promptText) {
                console.warn(`Coluna "${columnName}" não encontrada para q_id: ${ev.question_id}`);
                return null;
            }

            return {
                id: ev.evaluation_id,
                prompt: promptText,
                resposta_A: ev.response_A,
                resposta_B: ev.response_B,
                domain: ev.domain,
                language: languageValue // Saves the language with the default 'en' if it was empty
            };

        }).filter(Boolean);

        console.log(`Carregamento concluído. Total de amostras válidas: ${dataCombined.length}`);

        fillFilters();
        document.getElementById('loading-status').style.display = 'none';
        document.getElementById('start-btn').disabled = false;
        document.getElementById('start-btn').innerText = "Iniciar Anotação";

    } catch (error) {
        console.error("Falha ao carregar os arquivos CSV:", error);
        const status = document.getElementById('loading-status');
        status.style.color = 'red';
        status.innerText = "Erro ao carregar os dados. Verifique o console (F12).";
    }
}

function parseCsv(filePath) { return new Promise((resolve, reject) => { Papa.parse(filePath, { download: true, header: true, skipEmptyLines: true, complete: results => resolve(results.data), error: err => reject(err) }); }); }

function fillFilters() {
    const domainMap = {
        "General Knowledge": "Conhecimento Geral",
        "Technical": "Técnico (Programação)",
        "Creative": "Criatividade",
    };

    const domains = [...new Set(dataCombined.map(item => item.domain))].sort();
    const domainContainer = document.getElementById('domain-multiselect-container');
    domainContainer.innerHTML = ''; 

    domains.forEach(domain => {
        if(domain) {
            const optionEl = document.createElement('div');
            optionEl.classList.add('multiselect-option');
            optionEl.dataset.value = domain;

            optionEl.innerText = domainMap[domain] || domain; 
            
            optionEl.addEventListener('click', () => {
                optionEl.classList.toggle('selected');
            });
            domainContainer.appendChild(optionEl);
        }
    });

    const languageSelect = document.getElementById('language-select');
    languageSelect.innerHTML = '';

    const optionPt = new Option("Português", "pt-br");
    const optionEn = new Option("Inglês", "en");
    const optionTodos = new Option("Ambos", "todos");

    languageSelect.add(optionPt);
    languageSelect.add(optionEn);
    languageSelect.add(optionTodos);

    languageSelect.value = "pt-br";
}

function startAnnotationSession() {
    const selectedDomains = Array.from(document.querySelectorAll('#domain-multiselect-container .multiselect-option.selected'))
                                .map(el => el.dataset.value);
    const selectedLanguage = document.getElementById('language-select').value; // ex: "pt-br"

    const filteredSamples = dataCombined.filter(item => {
        const matchDomain = selectedDomains.length === 0 || selectedDomains.includes(item.domain);

        let matchLanguage = false;
        if (selectedLanguage === 'todos') {
            matchLanguage = true;
        } else {
            const itemLang = item.language || '';
            matchLanguage = itemLang.toLowerCase().startsWith(selectedLanguage.toLowerCase().split('-')[0]);
        }
        
        return matchDomain && matchLanguage;
    });
    
    samplesAvailable = filteredSamples.filter(item => !notesMadeIDs.has(item.id));
    
    console.log(`${samplesAvailable.length} amostras disponíveis para este usuário.`);
    prepareNewBatch();
}

function prepareNewBatch() {
    if (samplesAvailable.length === 0) { showScreen('all-done-container'); return; }
    for (let i = samplesAvailable.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [samplesAvailable[i], samplesAvailable[j]] = [samplesAvailable[j], samplesAvailable[i]];
    }
    batchCurrent = samplesAvailable.slice(0, BATCH_SIZE);
    CurrentBatchIndex = 0;
    showScreen('annotation-container');
    loadNextSample();
}
function loadNextSample() {
    window.scrollTo(0, 0);

    if (CurrentBatchIndex >= batchCurrent.length) {
        document.getElementById('stats-gerais').innerText = `Você já anotou um total de ${notesMadeIDs.size} amostras.`;
        showScreen('batch-complete-container');
        return;
    }

    const item = batchCurrent[CurrentBatchIndex];
    document.getElementById('prompt').innerText = "Prompt: " + item.prompt;
    document.getElementById('response-a').innerText = item.resposta_A;
    document.getElementById('response-b').innerText = item.resposta_B;
    document.getElementById('progress').innerText = `Amostra ${CurrentBatchIndex + 1} de ${batchCurrent.length} (neste lote)`;
}
function saveAnnotation(escolha) {
    const currentItem = batchCurrent[CurrentBatchIndex];
    const dadosParaSalvar = { evaluation_id: currentItem.id, human_choice: escolha, human_id: humanId };
    fetch(SCRIPT_URL, { method: 'POST', mode: 'no-cors', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(dadosParaSalvar) })
        .catch(error => console.error("Erro ao salvar:", error));
    notesMadeIDs.add(currentItem.id);
    localStorage.setItem('notesMadeIDs', JSON.stringify(Array.from(notesMadeIDs)));
    samplesAvailable = samplesAvailable.filter(item => item.id !== currentItem.id);
    CurrentBatchIndex++;
    loadNextSample();
}