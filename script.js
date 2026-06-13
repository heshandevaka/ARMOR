document.addEventListener('DOMContentLoaded', () => {
    // 1. Navbar Scroll Effect
    const navbar = document.getElementById('navbar');
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // 2. Smooth Scrolling for Navigation Links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // 3. Scroll Reveal Animation using Intersection Observer
    const revealElements = document.querySelectorAll('.reveal');
    
    const revealOptions = {
        threshold: 0.15,
        rootMargin: "0px 0px -50px 0px"
    };
    
    const revealOnScroll = new IntersectionObserver(function(entries, observer) {
        entries.forEach(entry => {
            if (!entry.isIntersecting) {
                return;
            } else {
                entry.target.classList.add('active');
                // Optional: Stop observing once revealed
                observer.unobserve(entry.target);
            }
        });
    }, revealOptions);
    
    revealElements.forEach(el => {
        revealOnScroll.observe(el);
    });

    // 4. Agentic Chat Integration
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatSubmit = document.getElementById('chat-submit');
    const btnText = chatSubmit.querySelector('.btn-text');
    const btnLoader = chatSubmit.querySelector('.btn-loader');
    
    const chatError = document.getElementById('chat-error');
    const chatResponse = document.getElementById('chat-response');
    
    const responseDomains = document.getElementById('response-domains');
    const responseSuccinctTitle = document.getElementById('response-succinct-title');
    const responseSuccinct = document.getElementById('response-succinct');
    const responseDetailedContainer = document.getElementById('response-detailed-container');
    const responseDetailed = document.getElementById('response-detailed');
    const responseReferencesContainer = document.getElementById('response-references-container');
    const responseReferences = document.getElementById('response-references');
    if (chatForm) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const question = chatInput.value.trim();
            if (!question) return;

            // Gather selected domains
            const domainCheckboxes = document.querySelectorAll('input[name="domain"]:checked');
            const domains = Array.from(domainCheckboxes).map(cb => cb.value);

            // Set loading state
            chatSubmit.disabled = true;
            btnText.classList.add('hidden');
            btnLoader.classList.remove('hidden');
            
            chatError.classList.add('hidden');
            chatResponse.classList.add('hidden');
            
            try {
                // Determine API URL (default to localhost:18000 for local dev, or the production URL)
                const apiUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
                    ? 'http://localhost:18000/ask'
                    : 'https://armor-agent.duckdns.org/ask';
                
                const requestBody = {
                    question: question,
                    domains: domains
                };

                const response = await fetch(apiUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`API returned ${response.status}: ${errorText}`);
                }

                const data = await response.json();
                
                const isAbstained = data.abstained === true;
                const hasDomains = data.selected_domains && data.selected_domains.length > 0;

                // Populate Domains
                responseDomains.innerHTML = '';
                if (hasDomains) {
                    data.selected_domains.forEach(d => {
                        const span = document.createElement('span');
                        span.className = 'route-tag';
                        span.textContent = d.toUpperCase();
                        responseDomains.appendChild(span);
                    });
                } else {
                    const span = document.createElement('span');
                    span.className = 'route-tag neutral';
                    span.textContent = 'No matching telecom domain';
                    responseDomains.appendChild(span);
                }
                
                // Determine Support Label
                let supportLabel = "";
                if (isAbstained) {
                    supportLabel = "No corpus-supported answer";
                } else if (data.evidence_status === "missing") {
                    supportLabel = "Missing evidence";
                } else if (data.evidence_status === "low_confidence") {
                    supportLabel = "Low-confidence evidence";
                } else if (data.evidence_status === "partial") {
                    supportLabel = "Partial support";
                } else if (data.evidence_status === "strong") {
                    supportLabel = "Strong support";
                } else {
                    supportLabel = data.evidence_status || "";
                }

                if (supportLabel) {
                    const statusSpan = document.createElement('span');
                    statusSpan.className = isAbstained ? 'route-tag neutral' : 'route-tag';
                    if (!isAbstained) {
                        statusSpan.style.backgroundColor = 'var(--accent)';
                        statusSpan.style.color = 'var(--text-main)';
                    }
                    statusSpan.textContent = supportLabel.toUpperCase();
                    responseDomains.appendChild(statusSpan);
                }

                // Populate Answers
                responseSuccinctTitle.textContent = isAbstained ? "Corpus Coverage" : "High-Level Answer";
                responseSuccinct.textContent = data.succinct_answer || data.answer || "No answer provided.";
                
                if (isAbstained) {
                    responseDetailedContainer.classList.add('hidden');
                } else {
                    responseDetailedContainer.classList.remove('hidden');
                    responseDetailed.textContent = data.detailed_answer || data.answer || "No detailed answer provided.";
                }
                
                // Populate References
                responseReferences.innerHTML = '';
                const allReferences = data.references || [];

                if (allReferences.length > 0) {
                    responseReferencesContainer.classList.remove('hidden');
                    allReferences.forEach((ref, idx) => {
                        const article = document.createElement('div');
                        article.className = 'evidence-item';
                        
                        const title = document.createElement('div');
                        title.style.fontWeight = '700';
                        title.style.color = 'var(--text-main)';
                        title.style.marginBottom = '0.5rem';
                        title.textContent = `[${idx + 1}] ${ref.title || ref.doc_id}`;
                        
                        const meta = document.createElement('div');
                        meta.className = 'evidence-meta';
                        meta.innerHTML = `<span>${ref.source_type || 'Unknown'} &middot; ${ref.category || 'Unknown'}</span>`;
                        
                        article.appendChild(title);
                        article.appendChild(meta);

                        if (ref.link) {
                            const link = document.createElement('a');
                            link.className = 'reference-link';
                            link.href = ref.link;
                            link.target = '_blank';
                            link.rel = 'noreferrer';
                            link.textContent = ref.link;
                            article.appendChild(link);
                        } else {
                            const code = document.createElement('code');
                            code.style.fontSize = '0.85rem';
                            code.style.color = 'var(--primary)';
                            code.textContent = ref.doc_id;
                            article.appendChild(code);
                        }
                        
                        responseReferences.appendChild(article);
                    });
                } else {
                    responseReferencesContainer.classList.add('hidden');
                }

                
                // Show response
                chatResponse.classList.remove('hidden');
                
            } catch (err) {
                console.error(err);
                chatError.textContent = `Error connecting to Agent API. Please ensure the SSH tunnel is active. Details: ${err.message}`;
                chatError.classList.remove('hidden');
            } finally {
                // Reset loading state
                chatSubmit.disabled = false;
                btnText.classList.remove('hidden');
                btnLoader.classList.add('hidden');
            }
        });
    }
});
