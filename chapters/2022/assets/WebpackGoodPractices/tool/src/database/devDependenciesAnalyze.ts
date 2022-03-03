import { Model, DataTypes } from "sequelize";
import { db } from "./database";

export interface DevDependenciesAnalyzeAttributes {
    //common
    id: number;
    
    //stats
    quantityOfDependencies: number;
    quantityOfDevDependencies: number;

    quantityOfTargetDependencies: number;
    quantityOfWrongDevDependencies: number;
}
export class DevDependenciesAnalyze extends Model<DevDependenciesAnalyzeAttributes> {
    declare id: number;
    declare quantityOfDependencies: number;
    declare quantityOfDevDependencies: number;
    
    declare quantityOfTargetDependencies: number;
    declare quantityOfWrongDevDependencies: number;

}
DevDependenciesAnalyze.init({
    id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
    },
    quantityOfDependencies: DataTypes.INTEGER,
    quantityOfDevDependencies: DataTypes.INTEGER,

    quantityOfTargetDependencies: DataTypes.INTEGER,
    quantityOfWrongDevDependencies: DataTypes.INTEGER,
}, { sequelize: db, modelName: 'DevDependenciesAnalyze' });

export function saveDevDependenciesAnalyze(devDependenciesAnalyzeAttributes: DevDependenciesAnalyzeAttributes): Promise<DevDependenciesAnalyze> {
    return DevDependenciesAnalyze.create(devDependenciesAnalyzeAttributes);
}

export function clearAllDevDependenciesAnalyze(): Promise<number> {
    return DevDependenciesAnalyze.destroy({ where: {} });
}