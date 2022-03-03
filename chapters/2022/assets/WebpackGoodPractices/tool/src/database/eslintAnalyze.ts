import { Model, DataTypes } from "sequelize";
import { db } from "./database";

export interface EsLintAnalyzeAttributes {
    //common
    id: number;
    //stats
    eslintFiles: boolean;
    quantityOfPlugins: number;
    quantityOfRules: number;
    error: number;
}
export class EsLintAnalyze extends Model<EsLintAnalyzeAttributes> {
    declare id: number;
    declare eslintFiles: boolean;
    declare quantityOfPlugins: number;
    declare quantityOfRules: number;
    declare error: number;
}
EsLintAnalyze.init({
    id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
    },
    eslintFiles: DataTypes.BOOLEAN,
    quantityOfPlugins: DataTypes.INTEGER,
    quantityOfRules: DataTypes.INTEGER,
    error: DataTypes.INTEGER,
}, { sequelize: db, modelName: 'EsLintAnalyze' });

export function saveEsLintAnalyzeAttributes(esLintAnalyzeAttributesAttribut: EsLintAnalyzeAttributes): Promise<EsLintAnalyze> {
    return EsLintAnalyze.create(esLintAnalyzeAttributesAttribut);
}

export function clearAllEsLintAnalyze(): Promise<number> {
    return EsLintAnalyze.destroy({ where: {} });
}