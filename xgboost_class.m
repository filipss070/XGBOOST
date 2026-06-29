
sciezka_do_folderu = 'C:\Users\filip\Documents\PCA\dane_trening';
lista_plikow = dir(fullfile(sciezka_do_folderu, '*.mat'));

for i = 1:length(lista_plikow)
    pelna_sciezka = fullfile(sciezka_do_folderu, lista_plikow(i).name);
    load(pelna_sciezka);
end

log_cena_BTC = log(macierz_cen(:,5));
log_zwroty_1h_BTC1 = diff(log_cena_BTC(24*365*2+1:24*365*2+180*24));
log_zwroty_1h_BTC = diff(log_cena_BTC(24*35-1-1:24*365*2-1));

log_zwroty1 = diff(log_waluta_estymowana(24*365*2+2:24*365*2+180*24+1));
log_zwroty_1h1 = diff(log_waluta_estymowana(24*365*2+1:24*365*2+180*24));
log_zwroty_6h1 = log_waluta_estymowana(24*365*2+2:24*365*2+180*24) - log_waluta_estymowana(24*365*2+2-6:24*365*2+180*24-6);
log_zwroty_12h1 = log_waluta_estymowana(24*365*2+2:24*365*2+180*24) - log_waluta_estymowana(24*365*2+2-12:24*365*2+180*24-12);
adf_2tyg1 = adf_eg1(24*365*2+2:24*365*2+180*24);
adf_1tyg1 = adf_eg2(24*365*2+2:24*365*2+180*24);
Q_diag_history1 = zeros(24*365*2+180*24 - (24*365*2+2) + 1,liczba_komponentow+1);
Q_trace_history1 = zeros(24*365*2+180*24 - (24*365*2+2) + 1,1);
volatility = zeros(24*365*2-1 - (24*35-1) + 1,1);
volatility1 = zeros(24*365*2+180*24 - (24*365*2+2) + 1,1);
for i = 24*365*2+2:24*365*2+180*24
    volatility1(i-(24*365*2+2-1)) = std(diff(log_waluta_estymowana(i-24:i)));
    Q_diag_history1(i-(24*365*2+2-1),:) = diag(Q_history(:,:,i))';
    Q_trace_history1(i-(24*365*2+2-1)) = trace(Q_history(:,:,i))';
end
for i = 24*35-1:24*365*2-1
    volatility(i-(24*35-1-1)) = std(diff(log_waluta_estymowana(i-24:i)));
end
R_in_history1 = R_history(24*365*2+2:24*365*2+180*24);
z_score_in_history1 = z_score_history(24*365*2+2:24*365*2+180*24);
x_in_history1 = x_history(:,24*365*2+2:24*365*2+180*24)';

godziny1 = hour(time_axis(24*365*2+2:24*365*2+180*24));
dni_tyg1 = mod(weekday(time_axis(24*365*2+2:24*365*2+180*24)) - 2, 7);

godziny_sin1 = sin(2 * pi * godziny1 / 24);
godziny_cos1 = cos(2 * pi * godziny1 / 24);

dni_sin1 = sin(2 * pi * dni_tyg1 / 7);
dni_cos1 = cos(2 * pi * dni_tyg1 / 7);

syntetyczne_zwroty_history_in1 = syntetyczne_zwroty_history(:,24*365*2+2:24*365*2+180*24)';

X_train = [log_zwroty_1h(:) log_zwroty_12h(:) log_zwroty_6h(:) adf_1tyg(:) adf_2tyg(:) Q_diag_history Q_trace_history(:) R_in_history(:) z_score_in_history(:) x_in_history godziny_cos(:) godziny_sin(:) dni_cos(:) dni_sin(:) syntetyczne_zwroty_history_in volatility(:) log_zwroty_1h_BTC(:)];

prog = 0.002;
y_train = ones(length(log_zwroty),1);

for i = 1:length(y_train)
    if abs(log_zwroty(i)) > prog
        y_train(i) = y_train(i) + log_zwroty(i)/abs(log_zwroty(i));
    end
end

y_test = ones(length(log_zwroty1),1);

for i = 1:length(y_test)
    if abs(log_zwroty1(i)) > abs(prog)
        y_test(i) = y_test(i) + log_zwroty1(i)/abs(log_zwroty1(i));
    end
end


% Import modułu numpy do MATLABa
np = py.importlib.import_module('numpy');

log_zwroty = log_zwroty*100;
% Konwersja na typy zrozumiałe dla Pythona
X_train_py = np.array(X_train);
y_train_py = np.array(int32(y_train));

% Import XGBoost
xgb = py.importlib.import_module('xgboost');


disp('Trenowanie klasyfikatora 3-klasowego...');
model = xgb.XGBClassifier(...
    pyargs('n_estimators', int32(300), ...
           'max_depth', int32(3), ...            
           'learning_rate', 0.05, ...             
           'min_child_weight', int32(15), ...     
           'gamma', 0.5, ...                       
           'reg_lambda', 0, ...                  
           'objective', 'multi:softprob', ... 
           'num_class', int32(3), ...         
           'subsample', 0.8, ...
           'colsample_bytree', 0.8));

model.fit(X_train_py, y_train_py);
disp('Model XGBoost został pomyślnie wytrenowany!');

disp('Pobieranie ważności cech (Feature Importance)...');

importances_py = model.feature_importances_;
importances = double(importances_py);

liczba_kolumn = size(X_train, 2);
nazwy_cech = compose('Kolumna %d', 1:liczba_kolumn); 

[posortowane_wagi, indeksy] = sort(importances, 'descend');
posortowane_nazwy = nazwy_cech(indeksy);

figure;
barh(posortowane_wagi);
set(gca, 'YTick', 1:liczba_kolumn, 'YTickLabel', posortowane_nazwy);
set(gca, 'YDir', 'reverse'); 
title('Ważność Cech w Modelu XGBoost');
xlabel('Zysk z podziału (Gain)');
grid on;

X_test = [log_zwroty_1h1(:) log_zwroty_12h1(:) log_zwroty_6h1(:) adf_1tyg1(:) adf_2tyg1(:) Q_diag_history1 Q_trace_history1(:) R_in_history1(:) z_score_in_history1(:) x_in_history1 godziny_cos1(:) godziny_sin1(:) dni_cos1(:) dni_sin1(:) syntetyczne_zwroty_history_in1 volatility1(:) log_zwroty_1h_BTC1(:)];


X_test_py = np.array(X_test);

prob_py = model.predict_proba(X_test_py);
prob_matlab = double(prob_py);

%(Kolumna 1 = klasa 0, Kolumna 2 = klasa 1, Kolumna 3 = klasa 2)
szansa_spadek = prob_matlab(:, 1);
szansa_szum   = prob_matlab(:, 2);
szansa_wzrost = prob_matlab(:, 3);

y_pred_py = model.predict(X_test_py);

y_pred = double(y_pred_py)'; 

accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('Dokładność modelu: %.4f\n', accuracy);

classificationError = sum(y_pred ~= y_test);
fprintf('Błąd klasyfikacji: %d\n', classificationError);

figure();
plot(y_pred,'r');
hold on;
plot(y_test,'b');

figure();
plot(prob_matlab(:,1),'b');
hold on;
plot(prob_matlab(:,3),'r');

kapital = 10000;
kapital_history = zeros(length(y_test),1);

for i = 1:length(y_test)
    if szansa_wzrost(i) > 0.6
        kapital = kapital + kapital*(log_zwroty1(i) - 0.001);
    elseif szansa_spadek(i) > 0.6
        kapital = kapital + kapital*(-log_zwroty1(i) - 0.001); 
    end
    kapital_history(i) = kapital;
end

figure();
plot(kapital_history);
%{
% Obliczenie błędu RMSE w MATLABie
rmse = sqrt(mean((log_zwroty1 - y_pred/100).^2));
fprintf('Błąd RMSE: %.4f\n', rmse);
mae = mean(abs(log_zwroty1 - y_pred/100));
fprintf('Błąd MAE: %.4f\n', mae);
%}