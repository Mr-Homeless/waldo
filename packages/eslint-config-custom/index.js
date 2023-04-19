module.exports = {
  extends: [
    'plugin:@typescript-eslint/recommended',
    'plugin:prettier/recommended',
    'eslint:recommended',
    'next',
    'turbo',
    'prettier',
  ],
  plugins: ['prettier', 'jest'],
  env: {
    es2017: true,
    node: true,
  },
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 12,
    sourceType: 'module',
  },
  rules: {
    'react/jsx-key': 'off',
    'arrow-parens': ['error', 'as-needed'],
    'prettier/prettier': 'error',
    'no-trailing-spaces': 'error',
    'import/extensions': 'off',
    'no-unused-vars': 'off',
    'max-len': [
      'error',
      {
        code: 100,
        tabWidth: 2,
        ignoreComments: true,
        ignoreStrings: true,
      },
    ],
  },
};
