function [data, artifacts] = reject_artifacts(dataset, data)

[data, art_car] = reject_cars(dataset, data);

[data, art_jumps] = reject_jumps(dataset, data);
[data, art_muscle] = reject_muscle(dataset, data);
[data, art_eye] = reject_eye(dataset, data);
artifacts.car = art_car;
artifacts.jumps = art_jumps;
artifacts.muscle = art_muscle;
artifacts.eye = art_eye;

