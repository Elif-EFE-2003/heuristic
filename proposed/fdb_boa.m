% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %

function [best_fitness, best_solution, curve, population_history, fitness_history] = fdb_boa(problem)
    
    % --- Parametre Çekme ---
    dim = problem.dimension;       
    lb = problem.lb;              
    ub = problem.ub;              
    maxFE = problem.maxFe;        
    
    % --- HİPERPARAMETRE OPTİMİZASYONU ---
    n = 100;                      % Fark 1: Popülasyon 100 yapıldı (Daha yüksek zeka kapasitesi) [cite: 896]
    p = 0.5;                      % Fark 2: Sabit 0.8 yerine 0.5 ile Global/Lokal dengesi sağlandı [cite: 908]
    power_exponent = 0.1;         
    sensory_modality = 0.01;      
    
    FE = 0;                           
    curve = zeros(1, maxFE);          
    
    % --- Kayıt Mekanizması ---
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, n, dim);
    fitness_history = zeros(history_size, n);
    history_index = 1;
    
    % --- Başlatma ---
    Sol = initialization(n, dim, ub, lb);
    [Fitness, FE] = calculate_fitness(Sol', problem, FE);
    
    [fmin, I] = min(Fitness);
    best_pos = Sol(I, :);
    S = Sol;
    
    % İlk popülasyonu kaydet
    for eval_count = 1:n
        curve(eval_count) = fmin;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Sol, Fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Ana döngü iterasyon hesabı
    N_iter = ceil((maxFE - n) / (n * 2));  
    
    for t = 1:N_iter
        
        % --- FARK 3: FDB SEÇİM STRATEJİSİ ---
        % Rastgele bir kelebek seçmek yerine FDB kullanarak; hem fitness'ı iyi
        % hem de sürüden uzak olan "akıllı" bir rehber seçiyoruz[cite: 547].
        fdb_index = fitnessDistanceBalance(Sol, Fitness);
        
        for i = 1:n
            Fnew = Fitness(i);
            FP = sensory_modality * (Fnew^power_exponent);
            
            if rand < p
                % --- Global Arama (En iyiye uçuş) ---
                dis = rand * rand * best_pos - Sol(i, :);
                S(i, :) = Sol(i, :) + dis * FP;
            else
                % --- FARK 4: TURNUVA SEÇİMİ (Selection Strategy) ---
                % Lokal aramada rastgele adaylar yerine, sürüden rastgele iki aday
                % çekip birbirleriyle yarıştırıyoruz. Daha iyisini rehber alıyoruz[cite: 727].
                tour1 = ceil(rand * n);
                tour2 = ceil(rand * n);
                if Fitness(tour1) < Fitness(tour2)
                    JK2 = tour1;
                else
                    JK2 = tour2;
                end
                
                % fdb_index ile aynı olmamasını sağla (Hata önleyici)
                while JK2 == fdb_index || JK2 == i
                    JK2 = ceil(rand * n);
                end
                
                epsilon = rand;
                
                % --- FARK 5: HİBRİTLEME (DE Mutation Entegrasyonu) ---
                % Diferansiyel Gelişim algoritmalarından ilham aldık. Formüle
                % ufak bir kaotik mutasyon vektörü ekleyerek kelebeklerin
                % çukurlarda (yerel minimum) hapsolmasını engelledik[cite: 729, 907].
                mutation_vector = (rand(1, dim) - 0.5) .* (ub - lb) * 0.01;
                dis = epsilon * epsilon * Sol(fdb_index, :) - Sol(JK2, :) + mutation_vector;
                
                S(i, :) = Sol(i, :) + dis * FP;
            end
            
            % Sınır Kontrolü
            S(i, :) = bound(S(i, :), ub, lb);
        end
        
        % Yeni çözümleri değerlendir
        [Fnew_array, FE] = calculate_fitness(S', problem, FE);
        
        % Gelişim varsa güncelle
        for i = 1:n
            if Fnew_array(i) <= Fitness(i)
                Sol(i, :) = S(i, :);
                Fitness(i) = Fnew_array(i);
            end
            if Fnew_array(i) <= fmin
                best_pos = S(i, :);
                fmin = Fnew_array(i);
            end
        end
        
        % Geçmişi kaydet
        for eval_idx = 1:n
            eval_count = FE - n + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = fmin;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Sol, Fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        % Duyusal modalite güncelleme
        sensory_modality = sensory_modality_NEW(sensory_modality, N_iter);
        
        if FE >= maxFE
            break;
        end
    end
    
    % Eksik kalan eğriyi doldur
    for i = FE+1:maxFE
        curve(i) = fmin;
    end
    
    best_fitness = fmin;
    best_solution = best_pos;
end

% --- Yardımcı Fonksiyonlar ---
function Positions = initialization(popsize, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(popsize, dim) .* (ub - lb) + lb;
    else
        for i = 1:dim
            ub_i = ub(i); lb_i = lb(i);
            Positions(:, i) = rand(popsize, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

function y = sensory_modality_NEW(x, Ngen)
    y = x + (0.025 / (x * Ngen));
end