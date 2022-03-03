module.exports = {
  env: {
    es2021: true,
    node: true,
  },
  extends: [
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: './tsconfig.json',
    exclude: [
      'node_modules',
      'repositories',
      '.eslintrc.js',
      '*.js',
    ],
  },
  plugins: [
    '@typescript-eslint',
  ],
  rules: {
    'eslint-disable no-shadow': 'off',
    'no-async-promise-executor': 'error',
    'no-await-in-loop': 'error',
    "@typescript-eslint/await-thenable": 2,
    "@typescript-eslint/ban-ts-comment": 2,
    "@typescript-eslint/camelcase": 0,
    "@typescript-eslint/class-name-casing": 0,
    "@typescript-eslint/interface-name-prefix": 0,
    "@typescript-eslint/member-delimiter-style": 0,
    "@typescript-eslint/no-empty-function": 0,
    "@typescript-eslint/no-empty-interface": 0,
    "@typescript-eslint/no-floating-promises": 2,
    "@typescript-eslint/no-inferrable-types": 0,
    "@typescript-eslint/no-namespace": 0,
    "@typescript-eslint/no-this-alias": 1,
    "@typescript-eslint/no-unused-vars": 0,     // does not work with class members 
    "@typescript-eslint/no-use-before-define": 0,
    "@typescript-eslint/no-var-requires": 0,
    "@typescript-eslint/type-annotation-spacing": 0,

    "curly": ["warn", "all"],
    "comma-dangle": ["warn", "always-multiline"],
    "prefer-const": 1,
    "prefer-spread": 1,
    "prefer-rest-params": 1,
    "no-void": "error",
    "require-await": "error",
  },
};
