# LAB-226 — Escopo clínico, permissões e métricas

| Campo | Valor |
|---|---|
| **Card Jira** | LAB-226 (ref. CE-9718) |
| **Épico** | LAB-131 — Assistente Clínico com IA |
| **PRD** | PRD0010 (Confluence Engenharia) |
| **Owner** | Rodrigo Steffler Kohl |
| **Período planejado** | 13/05/2026 – 15/05/2026 |
| **Status** | PRONTO PARA DEV |
| **Entrega esperada** | Matriz de escopo, fora de escopo, permissões e eventos |
| **Marco** | Produto e engenharia alinhados antes da implementação de interface e IA (gate para LAB-227+) |

## 1. Decisão de produto (resumo)

- **Anna Assistant** permanece como assistente **global**, focada em suporte e navegação do sistema.
- **Assistente Clínico** é uma feature **separada**, disponível **apenas dentro do prontuário** quando há atendimento ativo, contextualizada ao paciente.
- Anna **não** responde dúvidas clínicas diretamente. Quando houver atendimento ativo, ela **encaminha** para o Assistente Clínico (LAB-232).
- Toda resposta declara: paciente ativo, dados utilizados e que se trata de **apoio à decisão**, não diagnóstico autônomo.

## 2. Escopo clínico do MVP

Tipos de dúvida que o Assistente Clínico **deve** atender no MVP, sempre contextualizadas ao paciente ativo e ao módulo atual do atendimento:

| # | Tipo de dúvida | Exemplo | Módulo de origem típico |
|---|---|---|---|
| E1 | **Interpretação de achados** | "O que esses valores de TSH e T4 livre sugerem nesse paciente?" | Análise de exames |
| E2 | **Confirmação de conduta** | "A dose de losartana 50 mg/dia é adequada para esse perfil?" | Prescrição |
| E3 | **Avaliação de risco** | "Há risco de interação entre a prescrição atual e o histórico relatado?" | Prescrição / Anamnese |
| E4 | **Dados faltantes para decisão** | "O que ainda falta avaliar para fechar essa hipótese?" | Anamnese / Avaliação |
| E5 | **Resumo clínico do contexto** | "Resuma os achados relevantes do paciente até aqui." | Qualquer módulo do atendimento |
| E6 | **Educação clínica pontual** | "Quais critérios atuais para diagnóstico de síndrome metabólica?" | Qualquer módulo |

> Toda resposta segue o formato estruturado definido em **LAB-230** (achados, sugestões, pontos de atenção, limites, aviso final).

## 3. Fora de escopo do MVP

O Assistente Clínico **não** deve, no MVP:

| # | Item | Tratativa |
|---|---|---|
| F1 | Diagnóstico autônomo ou definitivo | Resposta sempre como apoio à decisão; decisão final é do profissional |
| F2 | Prescrição automática (sem revisão humana) | Sugestões são copiáveis/adicionáveis ao prontuário só com ação explícita (LAB-231) |
| F3 | Dúvidas operacionais / navegação do sistema | Permanecem com a Anna Assistant global |
| F4 | Uso fora do atendimento ativo | Entry point indisponível sem paciente em atendimento (LAB-227) |
| F5 | Anexar/processar arquivos enviados pelo usuário (PDFs, imagens, exames externos) | Fora do MVP — só dados já no prontuário entram no contexto |
| F6 | Memória entre atendimentos diferentes | Cada sessão usa apenas o contexto do atendimento atual |
| F7 | Conversa com a IA sobre **outro** paciente que não o ativo | Bloqueado pelo contrato de contexto (LAB-229) |
| F8 | Geração de laudos finais | Apenas auxílio na redação, sob revisão humana |

## 4. Permissões por perfil

> **Nota:** os perfis abaixo derivam dos usuários listados no PRD0010 §4 (Médicos, Dentistas, Profissionais de estética, Outros profissionais de saúde). A granularidade exata de roles do sistema Clínica Experts precisa ser validada com o time de produto. **Itens marcados com (?) precisam de confirmação.**

| Perfil | Abrir Assistente | Enviar dúvida | Copiar resposta | Adicionar resposta ao prontuário | Ver histórico da sessão |
|---|:---:|:---:|:---:|:---:|:---:|
| **Médico** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Dentista** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Profissional de estética** | ✅ | ✅ | ✅ | ✅ (?) | ✅ |
| **Outros profissionais de saúde** (enfermagem, nutrição, fisio, psicologia) | ✅ | ✅ | ✅ | ✅ (?) | ✅ |
| **Secretária / Recepção** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Administrador da clínica (não-clínico)** | ❌ | ❌ | ❌ | ❌ | ❌ |

**Regras transversais:**
- Acesso ao Assistente requer um **atendimento ativo** no momento da abertura (LAB-227).
- Adicionar resposta ao prontuário exige **confirmação explícita** do profissional (LAB-231) e fica marcada como "origem: IA — revisada por <profissional>".
- O contexto enviado à IA contém apenas dados do **paciente do atendimento ativo** (LAB-229).

## 5. Eventos de telemetria

Nomenclatura proposta: `clinical_assistant.<event>` — payload base inclui `user_id`, `user_role`, `attendance_id`, `patient_id_hash`, `module`, `timestamp`.

| Categoria | Evento | Quando dispara | Propriedades adicionais |
|---|---|---|---|
| **Entry point** | `clinical_assistant.entrypoint_shown` | Botão/atalho aparece para o profissional | `module` |
| | `clinical_assistant.entrypoint_clicked` | Profissional clica para abrir | `module` |
| **Drawer** | `clinical_assistant.drawer_opened` | Drawer abre com sucesso | `source` (entrypoint / anna_redirect) |
| | `clinical_assistant.drawer_closed` | Drawer fechado | `duration_ms`, `messages_sent` |
| **Interação** | `clinical_assistant.question_sent` | Profissional envia pergunta | `question_length`, `has_suggested_template` |
| | `clinical_assistant.suggestion_used` | Profissional usa uma sugestão guiada | `suggestion_id` |
| | `clinical_assistant.response_received` | IA retorna resposta estruturada | `latency_ms`, `response_sections`, `context_fields_used` |
| | `clinical_assistant.response_error` | Falha na resposta | `error_code`, `latency_ms` |
| **Ações sobre resposta** | `clinical_assistant.response_copied` | Profissional copia trecho | `section` (achados/sugestoes/etc) |
| | `clinical_assistant.response_added_to_record` | Resposta adicionada ao prontuário (após revisão) | `section`, `edited_before_save` (bool) |
| **Feedback** | `clinical_assistant.feedback_submitted` | Usuário avalia resposta (👍/👎 + texto) | `rating`, `has_comment` |
| **Encaminhamento Anna** | `clinical_assistant.referred_from_anna` | Anna redirecionou dúvida clínica para o Assistente | `original_question_id` |
| **Abandono** | `clinical_assistant.session_abandoned` | Drawer fechado sem `question_sent` | `time_to_close_ms` |

**Indicadores derivados** (cálculo no BI, não eventos novos):
- Taxa de adoção = `entrypoint_clicked` / `entrypoint_shown` por perfil
- Taxa de abandono = `session_abandoned` / `drawer_opened`
- Recorrência = nº de profissionais com `question_sent` em ≥ 2 atendimentos distintos / nº total
- Conversão para prontuário = `response_added_to_record` / `response_received`
- Saída do sistema durante atendimento = (proxy) sessões sem `question_sent` mas com janela em foco perdida > X s — **TBD se vamos instrumentar isso no MVP**

## 6. Critérios de aceite (MVP)

Conforme cronograma e PDF de referência:

**Produto**
- Assistente disponível **apenas** no atendimento.
- Clareza entre IA clínica e Anna Assistant global.
- Onboarding explica limites e finalidade.

**Segurança**
- Contexto clínico limitado ao necessário (LAB-229).
- Resposta declara apoio à decisão, não diagnóstico autônomo.
- Escrita no prontuário depende de ação humana (LAB-231).

**Operação**
- Métricas de uso, abandono e feedback registradas (§5 deste doc).
- Cards rastreados pelas labels Jira `prd0010` e `assistente-clinico`.
- Fluxo pronto para validação controlada.

## 7. Riscos e pontos de atenção

| Risco | Mitigação no MVP |
|---|---|
| **Risco clínico/jurídico** — IA percebida como prescritora | Resposta sempre com aviso de apoio à decisão; sem escrita automática; revisão humana obrigatória |
| **Confusão com Anna Assistant** | Anna não responde clínica; encaminha quando há atendimento ativo (LAB-232); onboarding explicita o papel |
| **Contexto insuficiente** | Resposta declara explicitamente quais dados foram usados e quais campos estão faltando |
| **Adoção e confiança** | Sugestões guiadas + respostas estruturadas + feedback simples; comunicação inicial da feature |

## 8. Pontos abertos para validação

> Itens que não estavam totalmente fechados no PRD e precisam de decisão antes de fechar este card.

- [ ] Permissões finais por perfil — confirmar com produto se "Profissional de estética" e "Outros profissionais de saúde" podem **adicionar ao prontuário** ou apenas copiar (§4, marcadores `?`).
- [ ] Lista canônica de roles do sistema (mapear nomenclatura do Clínica Experts → perfis acima).
- [ ] Instrumentação de "saída do sistema durante atendimento" entra ou não no MVP (§5).
- [ ] Limite de tokens/turnos por sessão do drawer (governança de custo).
- [ ] Modelo de IA selecionado e fallback — **depende do benchmark deste repo** (`1_download_data.py` → `5_visualize.py`).
- [ ] **LLM as a judge** para avaliação de qualidade clínica — métricas atuais do `3_score.py` (bge-m3 dense/sparse, e5, BERTScore) medem similaridade lexical/semântica vs. referência, o que é insuficiente para julgar correção clínica, aderência ao formato estruturado do LAB-230 (achados / sugestões / pontos de atenção / limites / aviso final) e ausência de alucinação. Decidir: (a) modelo-juiz e rubrica (correção factual, completude, segurança, formato, tom de apoio à decisão); (b) se roda offline no benchmark, online sobre `response_received` amostrado, ou ambos; (c) calibração contra avaliação humana de uma amostra antes de adotar como gate de release.
- [ ] Política de retenção do histórico do chat clínico (sessão? atendimento? auditável por quanto tempo?).
- [ ] **Arquitetura: extensão da Anna Assistant ou projeto independente** — decidir se o Assistente Clínico será desenvolvido como extensão/módulo do projeto atual da Anna Assistant (reuso de infra, autenticação, UI do drawer, pipeline de prompt) ou como projeto independente (repositório, deploy e ciclo de release próprios). Impacta estimativa, ownership, versionamento e estratégia de release das LAB-227+.
- [ ] **Impacto do redesign do prontuário no entry point** — observação de João Libio (28/12/2025):
  > No relayout do prontuário, poderemos fazer anotações, preencher fichas diretamente no ambiente do paciente, neste caso, o funcionamento será o mesmo? O botão do assistente clínico, vai ficar disponível apenas na tela do prontuário/atendimento?
  >
  > A ideia do redesign do prontuário é torná-lo mais simples e intuitivo, onde não preciso criar um atendimento para isso, mas sim, já ir realizando as ações que o profissional da saúde necessita para a consulta/procedimento executado. Acredito que tenha impacto.
  >
  > Sugestão para avaliarmos: nesta tela (prontuário/atendimento) o profissional da saúde pode chamar a Anna Assistent, contudo, o que será aberto no drawer lateral, nesta tela sempre será o assistente clínico. Penso que devemos manter a feature da assistent com comportamento igual em todo o sistema.

  Afeta a regra "atendimento ativo" do §4 e o item F4 do §3 — revisitar com produto antes de fechar LAB-227.

---

_Documento de definição do card LAB-226. Gate para LAB-227 (entry point), LAB-228 (drawer), LAB-229 (contexto), LAB-230 (resposta estruturada), LAB-231 (copiar/adicionar), LAB-232 (encaminhamento Anna + onboarding + feedback)._
